import glob
import os
import random

import cv2
import numpy as np
import torch
from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide

from .data_processing import get_mask_from_json
from .refer import REFER
from .refer_seg_dataset import ReferSegDataset
from .reason_seg_dataset import ReasonSegDataset
from .qa_template import SHORT_ANSWER_TEMPLATE, SHORT_QUESTION_TEMPLATE, NEG_ANSWER_TEMPLATE, CORRECT_ANSWER_TEMPLATE, LONG_QUESTION_TEMPLATE, LONG_ANSWER_TEMPLATE
from .trainval_dataset import collate_fn_val
DEFAULT_IMAGE_TOKEN = "<image>"

collate_fn_test = collate_fn_val


class TestReferDataset(ReferSegDataset):
    def __init__(
        self,
        base_image_dir,
        vision_tower,
        image_size: int = 336,
        num_classes_per_sample: int = 1,
        train_test_split="val",
        datasetname="fprefcoco",
        use_val_mode=True,
        use_test_mode=False,
        conversation_records = None
    ):
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = vision_tower
        self.use_val_mode = use_val_mode
        self.use_test_mode = use_test_mode
        self.conversation_records = conversation_records

        self.short_question_list = SHORT_QUESTION_TEMPLATE
        self.answer_list = SHORT_ANSWER_TEMPLATE
        self.neg_answer_list = NEG_ANSWER_TEMPLATE
        self.correct_answer_list = CORRECT_ANSWER_TEMPLATE
        # Load dataset
        self.ds = ds = datasetname

        data_dir = os.path.join(self.base_image_dir, "refer_seg")
        split_by = self.determine_split_by(ds)
        refer_api = REFER(data_dir, ds, split_by)
        ref_ids_test = refer_api.getRefIds(split=train_test_split)
        images_ids_test = refer_api.getImgIds(ref_ids=ref_ids_test)
        refs_test = refer_api.loadRefs(ref_ids=ref_ids_test)
        self.test_dataset = self.prepare_dataset(ds, refer_api, images_ids_test, refs_test, data_dir)
        print("data length = ", len(self.test_dataset["images"]))

    def __len__(self):
        return len(self.test_dataset["images"])

    def select_dataset_and_image(self, idx):
        """Selects a random dataset and an image from it."""
        refer_seg_ds = self.test_dataset
        images, annotations, img2refs = refer_seg_ds["images"], refer_seg_ds["annotations"], refer_seg_ds["img2refs"]
        
        image_info = images[idx]
        image_id = image_info["id"]
        refs = img2refs[image_id]
        return self.ds, image_info, refs, annotations

    def process_referring_expressions(self, refs):
        # Load referring expression info.
        Q_sents = []
        gt_sents = []
        ann_ids = []
        ref_ids = []
        sent_ids = []
        exists = []
        for ref in refs:
            for idx, sent in enumerate(ref["sentences"]):
                text = sent["sent"]
                Q_sents.append(text)
                gt_sents.append(sent.get("gt_sent", ""))
                ann_ids.append(ref["ann_id"])
                ref_ids.append(ref["ref_id"])
                sent_ids.append(idx)
                if "is_false_premise" in sent:
                    exists.append(not sent["is_false_premise"])
                elif "exist" in sent:
                    exists.append(sent["exist"])
                else:
                    exists.append(True)
        return Q_sents, gt_sents, ann_ids, exists, ref_ids, sent_ids

    def __getitem__(self, idx):
        # get one sample
        ds, image_info, refs, annotations = self.select_dataset_and_image(idx)
        # Load images and clip features
        image, image_clip, sam_input_shape = self.load_and_preprocess_image(image_info["file_name"])
        # load referring expression
        Q_sents, A_sents, ann_ids, exists, ref_ids, sent_ids = self.process_referring_expressions(refs)
        # create conversation Q/A (convert it to LLaVA type)
        if self.use_val_mode:
            conversations = self.create_conversations(ds, Q_sents, A_sents, exists, load_answer=True)
        if self.use_test_mode: # for test mode
            conversations = self.create_conversations(ds, Q_sents, A_sents, exists, load_answer=False)
        # load segmentation masks
        masks = self.load_segmentation_masks(image_info, annotations, sam_input_shape, ann_ids, exists, include_nonexist=True)
        sam_mask_shape = [sam_input_shape, (masks.shape[1], masks.shape[2])]
        # print(masks.shape[1] == sam_mask_shape[2] and masks.shape[2] == sam_mask_shape[3], flush=True)
        conversation_record = {image_info["file_name"]:conversations}
        return (
            image_info["file_name"],    # filename
            image,                      # raw image (for SAM)
            image_clip,                 # image clip feature (for LMMs)
            conversations,              # QA
            masks,                      # segmentation GT
            sam_mask_shape,             # input / output shape for SAM
            exists,                     # object existence
            ref_ids,                    # ref id (useless now)
            sent_ids,                    # sent id (useless now)
            conversation_record
        )


class TestReasoningDataset(ReasonSegDataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255
    def __init__(
        self,
        base_image_dir,
        vision_tower,
        image_size: int = 336,
        num_classes_per_sample: int = 1,
        train_test_split="val",
        datasetname="ReasonSeg",
        use_val_mode=True,
        use_test_mode=False,
        eval_only = False,
        conversation_records = None
    ):
        self.image_size = image_size
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = vision_tower
        self.num_classes_per_sample = num_classes_per_sample
        self.base_image_dir = base_image_dir
        self.use_val_mode = use_val_mode
        self.use_test_mode = use_test_mode
        self.eval_only = eval_only
        
        DEFAULT_IMAGE_TOKEN = "<image>"
        SHORT_QUESTION_TEMPLATE = [
            DEFAULT_IMAGE_TOKEN
            + "\n"
            + "What is {class_name} in this image? Please output segmentation mask."
        ]
        LONG_QUESTION_TEMPLATE = [
            DEFAULT_IMAGE_TOKEN
            + "\n"
            + "{sent} Please output segmentation mask.",
        ]
        LONG_ANSWER_TEMPLATE = ["Sure, the segmentation result is [SEG]."]

        self.short_question_list = SHORT_QUESTION_TEMPLATE
        self.long_question_list = LONG_QUESTION_TEMPLATE
        self.answer_list = LONG_ANSWER_TEMPLATE
        self.conversation_records = conversation_records
        # load dataset
        reason_seg_data, splits = datasetname, train_test_split
        splits = splits.split("_")
        images = []
        for split in splits:
            images_split = glob.glob(
                os.path.join(
                    base_image_dir, "reason_seg", reason_seg_data, split, "*.jpg"
                )
            )
            images.extend(images_split)
        #for long query
        #images = [ i for i in images if get_mask_from_json(i.replace(".jpg", ".json"), 
        #                   cv2.cvtColor(cv2.imread(i), cv2.COLOR_BGR2RGB))[3]]
        # for short query
        #images = [ i for i in images if not get_mask_from_json(i.replace(".jpg", ".json"), 
        #                  cv2.cvtColor(cv2.imread(i), cv2.COLOR_BGR2RGB))[3]]
        jsons = [path.replace(".jpg", ".json") for path in images]
        self.reason_seg_data = (images, jsons)

        print("number of reason_seg samples: ", len(images))

    def __len__(self):
        return len(self.reason_seg_data[0])

    def __getitem__(self, idx):

        images, jsons = self.reason_seg_data
        image_path = images[idx]
        json_path = jsons[idx]

        # image_path = '../dataset/reason_seg/ReasonSeg/train/4971309080_370ab0baf3_o.jpg'
        # json_path = '../dataset/reason_seg/ReasonSeg/train/4971309080_370ab0baf3_o.json'
        # image_path = '../dataset/reason_seg/ReasonSeg/val/scene0104_00_0.jpg'
        # json_path = '../dataset/reason_seg/ReasonSeg/val/scene0104_00_0.json'
        
        # Load images and clip features
        image, image_clip, sam_input_shape = self.load_and_preprocess_image(image_path)
        # Get sents and segmentation maps
        img = cv2.imread(image_path)[:, :, ::-1]
        mask, sents, fp_qa, is_sentence = get_mask_from_json(json_path, img)
        # Sampling
        # assert len(sents) == len(fp_qa) == 1

        # Create Q/A Data
        conversations = []
        # True premise question
        conv = conversation_lib.default_conversation.copy()
        if is_sentence:
            question_template = random.choice(self.long_question_list)
            Q_sent = question_template.format(sent=sents[0])
        else:
            question_template = random.choice(self.short_question_list)
            Q_sent = question_template.format(class_name=sents[0].lower())
        conv.append_message(conv.roles[0], Q_sent)
        if self.use_val_mode:
            conv.append_message(conv.roles[1], random.choice(self.answer_list))
        if self.use_test_mode: 
            conv.append_message(conv.roles[1], None)
        conversations.append(conv.get_prompt())
        
        debug = False # set true to align Lisa
        if debug:
            sampled_sents = [sents[0]]
            conversations = []
            conv = conversation_lib.default_conversation.copy()
            i = 0
            while i < len(sampled_sents):
                conv.messages = []
                text = sampled_sents[i].strip()
                if is_sentence:
                    conv.append_message(
                        conv.roles[0],
                        DEFAULT_IMAGE_TOKEN
                        + "\n {} Please output segmentation mask.".format(text),
                    )
                    conv.append_message(conv.roles[1], "[SEG].")
                else:
                    conv.append_message(
                        conv.roles[0],
                        DEFAULT_IMAGE_TOKEN
                        + "\n What is {} in this image? Please output segmentation mask.".format(
                            text
                        ),
                    )
                    conv.append_message(conv.roles[1], "[SEG].")
                conversations.append(conv.get_prompt())
                i += 1

        conv = conversation_lib.default_conversation.copy()
        # False premise question
        if fp_qa is True:
            if fp_qa[0][1] is True:
                question_template = random.choice(self.long_question_list)
                neg_Q_sent = question_template.format(sent=fp_qa[0][0])
            else:
                question_template = random.choice(self.short_question_list)
                neg_Q_sent = question_template.format(class_name=fp_qa[0][0])
            conv.append_message(conv.roles[0], neg_Q_sent)

            if self.use_val_mode:
                conv.append_message(conv.roles[1], fp_qa[0][2])
            if self.use_test_mode:
                conv.append_message(conv.roles[1], None)
            conversations.append(conv.get_prompt())
        
        #if self.eval_only: 
        #    # only to restore template
        #    conversations = self.conversation_records[image_path.replace('dataset', 'dataset_sesame')]

        # Exists and segmentation masks
        conversation_record = {image_path:conversations}
        exists = [True, False] if fp_qa is True else [ True ]
        masks = [ mask, np.zeros_like(mask).astype(np.float32) ] if self.use_test_mode else [ mask ]
        masks = torch.from_numpy(np.stack(masks, axis=0))
        sam_mask_shape = [sam_input_shape, (masks.shape[1], masks.shape[2])]
        ref_ids = [int(idx), int(idx)]
        sent_ids = [0, 1]
        return (
            image_path,         # filename
            image,              # raw image (for SAM)
            image_clip,         # image clip feature (for LMMs)
            conversations,      # QA
            masks,              # segmentation GT
            sam_mask_shape,     # input / output shape for SAM
            exists,             # object existence
            ref_ids,            # ref id (useless now)
            sent_ids,            # sent id (useless now)
            conversation_record
        )
