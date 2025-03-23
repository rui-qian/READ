import os
import random
import numpy as np
import torch
from pycocotools import mask

from model.llava import conversation as conversation_lib
from .base_dataset import BaseDataset
from .refer import REFER
from .qa_template import SHORT_QUESTION_TEMPLATE, SHORT_ANSWER_TEMPLATE, NEG_ANSWER_TEMPLATE, CORRECT_ANSWER_TEMPLATE


def create_zero_mask(height, width):
    return np.zeros((height, width), dtype=np.uint8)


def decode_segmentation(ann_segmentation, height, width):
    if type(ann_segmentation[0]) == list:  # polygon
        rle = mask.frPyObjects(ann_segmentation, height, width)
    else:
        rle = ann_segmentation
        for seg in rle:
            if not isinstance(seg["counts"], bytes):
                seg["counts"] = seg["counts"].encode()
    masks = mask.decode(rle)
    return np.sum(masks, axis=2).astype(np.uint8)  # Convert to np.uint8


def process_annotation(ann, image_info):
    if len(ann["segmentation"]) == 0:
        return create_zero_mask(image_info["height"], image_info["width"])
    else:
        return decode_segmentation(ann["segmentation"], image_info["height"], image_info["width"])


class ReferSegDataset(BaseDataset):

    def __init__(
        self,
        base_image_dir,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        image_size: int = 336,
        num_classes_per_sample: int = 3,
        refer_seg_data="refclef||refcoco||refcoco+||refcocog",
        train_val_split="train",
    ):
        super().__init__(vision_tower, samples_per_epoch, image_size)
        self.num_classes_per_sample = num_classes_per_sample
        self.base_image_dir = base_image_dir

        self.short_question_list = SHORT_QUESTION_TEMPLATE
        self.answer_list = SHORT_ANSWER_TEMPLATE
        self.neg_answer_list = NEG_ANSWER_TEMPLATE
        self.correct_answer_list = CORRECT_ANSWER_TEMPLATE

        self.refer_seg_data = self.load_refer_seg_data(refer_seg_data, train_val_split)

    def load_refer_seg_data(self, refer_seg_data, train_val_split):
        """Loads the refer segmentation data."""
        data_dir = os.path.join(self.base_image_dir, "refer_seg")
        self.refer_seg_ds_list = refer_seg_data.split("||")
        refer_seg_data = {}
        for ds in self.refer_seg_ds_list:
            split_by = self.determine_split_by(ds)
            refer_api = REFER(data_dir, ds, split_by)
            ref_ids_train = refer_api.getRefIds(split=train_val_split)
            images_ids_train = refer_api.getImgIds(ref_ids=ref_ids_train)
            refs_train = refer_api.loadRefs(ref_ids=ref_ids_train)

            refer_seg_ds = self.prepare_dataset(ds, refer_api, images_ids_train, refs_train, data_dir)
            refer_seg_data[ds] = refer_seg_ds
        return refer_seg_data

    def determine_split_by(self, ds):
        """Determines the split type based on the dataset."""
        if ds == "refclef":
            return "unc"
        elif ds in ["refcocog", "R-refcocog"]:
            return "umd_exclude_unified"
        elif ds in ["fprefcocog", "fprefcoco", "fprefcoco+"]:
            return "berkeley_exclude_unified"
        return "unc_exclude_unified"

    def prepare_dataset(self, ds, refer_api, image_ids, refs, data_dir):
        """Prepares the dataset for a given segmentation data source."""
        refer_seg_ds = {"images": [], "annotations": refer_api.Anns}
        for item in refer_api.loadImgs(image_ids):
            item = item.copy()
            item["file_name"] = self.get_image_path(ds, item, data_dir)
            refer_seg_ds["images"].append(item)
        img2refs = {}
        for ref in refs:
            img2refs[ref["image_id"]] = img2refs.get(ref["image_id"], []) + [ref]
        refer_seg_ds["img2refs"] = img2refs

        print(f"Dataset {ds} (refs {self.determine_split_by(ds)}) (train split) has {len(refer_seg_ds['images'])} images and {len(refer_seg_ds['annotations'])} annotations.")
        return refer_seg_ds

    def get_image_path(self, ds, item, data_dir):
        """Returns the correct image path based on the dataset."""
        if ds == "refclef":
            return os.path.join(data_dir, "images/saiapr_tc-12", item["file_name"])
        return os.path.join(data_dir, "images/mscoco/images/train2014", item["file_name"])

    def select_dataset_and_image(self):
        """Selects a random dataset and an image from it."""
        ds = random.choice(self.refer_seg_ds_list)
        refer_seg_ds = self.refer_seg_data[ds]
        images, annotations, img2refs = refer_seg_ds["images"], refer_seg_ds["annotations"], refer_seg_ds["img2refs"]
        idx = random.randint(0, len(images) - 1)
        image_info = images[idx]
        image_id = image_info["id"]
        refs = img2refs[image_id]
        return ds, image_info, refs, annotations

    def process_referring_expressions(self, refs):
        # Load referring expression info.
        sents = []
        gt_sents = []
        ann_ids = []
        exists = []
        for ref in refs:
            for sent in ref["sentences"]:
                text = sent["sent"]
                sents.append(text)
                ann_ids.append(ref["ann_id"])
                gt_sents.append(sent.get("gt_sent", ""))
                if "is_false_premise" in sent:
                    exists.append(not sent["is_false_premise"])
                elif "exist" in sent:
                    exists.append(sent["exist"])
                else:
                    exists.append(True)
                # exists.append(not sent.get("is_false_premise", False) or sent.get("exist", True))
        sample_size = min(len(sents), self.num_classes_per_sample)
        sampled_inds = random.sample(range(len(sents)), sample_size) if len(sents) >= self.num_classes_per_sample else range(len(sents))
        # Sampling process
        sampled_Q_sents = [sents[ind] for ind in sampled_inds]
        sampled_ann_ids = [ann_ids[ind] for ind in sampled_inds]
        sampled_exists = [exists[ind] for ind in sampled_inds]
        sampled_A_sents = [gt_sents[ind] for ind in sampled_inds]
        return sampled_Q_sents, sampled_A_sents, sampled_ann_ids, sampled_exists

    def create_conversations(self, ds, Q_sents, A_sents, exists, load_answer=True):
        # Load conversations and Q/A
        conversations = []
        questions = []
        answers = []
        for idx, text in enumerate(Q_sents):

            question_template = random.choice(self.short_question_list)
            questions.append(question_template.format(class_name=text.lower()))

            corrected_sentence = A_sents[idx].strip()
            text = text.strip()
            assert len(text.split("||")) == 1

            if exists[idx] is True:
                answer_template = random.choice(self.answer_list)
                answers.append(answer_template.format(class_name=text.lower()))
            else:
                
                # false premise correction
                if ds in ["fprefcocog", "fprefcoco", "fprefcoco+"]:
                    answer_template = random.choice(self.correct_answer_list)
                    answers.append(
                        answer_template.format(
                            class_name=text.lower(), gt_name=corrected_sentence.lower()
                        )
                    )
                else:
                    answer_template = random.choice(self.neg_answer_list)
                    answers.append(answer_template.format(class_name=text.lower()))
        
            conv = conversation_lib.default_conversation.copy()
            conv.append_message(conv.roles[0], questions[idx])
            if load_answer is True:
                conv.append_message(conv.roles[1], answers[idx])
            else:
                conv.append_message(conv.roles[1], None)
            conversations.append(conv.get_prompt())
        return conversations

    def load_segmentation_masks(self, image_info, annotations, sam_input_shape, ann_ids, exists, include_nonexist=False):
        # Load segmentation masks
        masks = []
        for i, ann_id in enumerate(ann_ids):
            if include_nonexist is False and exists[i] is False:
                continue
            if isinstance(ann_id, list):
                combined_mask = create_zero_mask(image_info["height"], image_info["width"]) 
                if -1 not in ann_id: # valid annotations
                    for ann_id_i in ann_id:
                        combined_mask |= process_annotation(annotations[ann_id_i], image_info)
                m = combined_mask
            else:
                m = process_annotation(annotations[ann_id], image_info)
            # If include nonexist is True will also include a blank mask (for test usage)
            if exists[i] is False:
                m = np.zeros_like(m)
            masks.append(m)
        if len(masks) == 0:
            masks = np.zeros((0, *sam_input_shape))  # original input shape
        else:
            masks = np.stack(masks, axis=0)

        masks = torch.from_numpy(masks)
        return masks

    def __getitem__(self, idx):
        # get one sample
        ds, image_info, refs, annotations = self.select_dataset_and_image()
        # Load images and clip features
        image, image_clip, sam_input_shape = self.load_and_preprocess_image(image_info["file_name"])
        # load referring expression
        Q_sents, A_sents, ann_ids, exists = self.process_referring_expressions(refs)
        # create conversation Q/A (convert it to LLaVA type)
        conversations = self.create_conversations(ds, Q_sents, A_sents, exists)
        # load segmentation masks
        masks = self.load_segmentation_masks(image_info, annotations, sam_input_shape, ann_ids, exists)
        sam_mask_shape = [sam_input_shape, (masks.shape[1], masks.shape[2])]
        # print(masks.shape[1] == sam_mask_shape[2] and masks.shape[2] == sam_mask_shape[3], flush=True)
        return (
            image_info["file_name"],    # filename
            image,                      # raw image (for SAM)
            image_clip,                 # image clip feature (for LMMs)
            conversations,              # QA
            masks,                      # segmentation GT
            sam_mask_shape,             # input / output shape for SAM
            exists,                     # object existence
        )