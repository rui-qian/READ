import numpy as np
import torch
from model.segment_anything.utils.transforms import ResizeLongestSide

from .reason_seg_dataset import ReasonSegDataset
from .refer_seg_dataset import ReferSegDataset
from .sem_seg_dataset import SemSegDataset
from .vqa_dataset import VQADataset
from .qa_template import SHORT_ANSWER_TEMPLATE, SHORT_QUESTION_TEMPLATE, NEG_ANSWER_TEMPLATE
from .utils import replace_image_tokens, tokenize_and_pad, handle_conversation_specifics


def collate_fn_train(batch, tokenizer=None, conv_type="llava_v1", use_mm_start_end=True):
    image_path_list = []
    images_list = []
    images_clip_list = []
    conversation_list = []
    masks_list = []
    exists_list = []
    sam_mask_shape_list = []
    offset_list = [0]
    cnt = 0
    for (image_path, images, images_clip, conversations,
            masks, sam_mask_shape, exists) in batch:
        image_path_list.append(image_path)
        images_list.append(images)
        images_clip_list.append(images_clip)
        conversation_list.extend(conversations)
        masks_list.append(masks.float())
        sam_mask_shape_list.append(sam_mask_shape)
        cnt += len(conversations)
        offset_list.append(cnt)
        exists_list.append(exists)

    # Replace <image> token if use_mm_start_end is True
    if use_mm_start_end:
        conversation_list = replace_image_tokens(conversation_list)

    # Tokenization and padding of input IDs
    input_ids, attention_masks = tokenize_and_pad(conversation_list, tokenizer)

    # Generating targets (answer sentences) and handling conversation specifics
    targets = handle_conversation_specifics(input_ids, conversation_list, tokenizer, conv_type)

    # Truncate data if not in inference mode
    truncate_len = tokenizer.model_max_length - 255

    if input_ids.shape[1] > truncate_len:
        input_ids = input_ids[:, :truncate_len]
        targets = targets[:, :truncate_len]
        attention_masks = attention_masks[:, :truncate_len]

    return {
        "image_paths": image_path_list,
        "images": torch.stack(images_list, dim=0),
        "images_clip": torch.stack(images_clip_list, dim=0),
        "input_ids": input_ids,
        "labels": targets, # gt sentences (name compatible for HG pipeline)
        "attention_masks": attention_masks,
        "masks_list": masks_list,   # segmentation gt
        "sam_mask_shape_list": sam_mask_shape_list,
        "offset": torch.LongTensor(offset_list),
        "inference": False,
        "conversation_list": conversation_list,
        "exists": exists_list,
    }


class HybridDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_image_dir,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        image_size: int = 336,
        num_classes_per_sample: int = 3,
        dataset="sem_seg||refer_seg||vqa||reason_seg",
        sample_rate=[9, 3, 3, 1],
        sem_seg_data="ade20k||cocostuff||partimagenet||pascal_part||paco_lvis||mapillary",
        refer_seg_data="refclef||refcoco||refcoco+||refcocog",
        neg_refer_seg_data="R-refcoco||R-refcoco+||R-refcocog",
        correct_refer_seg_data="fprefcoco||fprefcoco+||fprefcocog",
        vqa_data="llava_instruct_150k",
        reason_seg_data="ReasonSeg|train",
    ):
        self.samples_per_epoch = samples_per_epoch
        sample_rate = np.array(sample_rate)
        self.sample_rate = sample_rate / sample_rate.sum()
        self.all_datasets = []
        for dataset in dataset.split("||"):
            if dataset == "sem_seg":
                self.all_datasets.append(
                    SemSegDataset(
                        base_image_dir,
                        vision_tower,
                        samples_per_epoch,
                        image_size,
                        num_classes_per_sample,
                        sem_seg_data,
                    )
                )
            elif dataset == "refer_seg":
                self.all_datasets.append(
                    ReferSegDataset(
                        base_image_dir,
                        vision_tower,
                        samples_per_epoch,
                        image_size,
                        num_classes_per_sample,
                        refer_seg_data,
                    )
                )
            elif dataset == "neg_refer_seg":
                self.all_datasets.append(
                    ReferSegDataset(
                        base_image_dir,
                        vision_tower,
                        samples_per_epoch,
                        image_size,
                        num_classes_per_sample,
                        neg_refer_seg_data,
                    )
                )
            elif dataset == "correct_refer_seg":
                self.all_datasets.append(
                    ReferSegDataset(
                        base_image_dir,
                        vision_tower,
                        samples_per_epoch,
                        image_size,
                        num_classes_per_sample,
                        correct_refer_seg_data,
                    )
                )
            elif dataset == "vqa":
                self.all_datasets.append(
                    VQADataset(
                        base_image_dir,
                        vision_tower,
                        samples_per_epoch,
                        image_size,
                        vqa_data,
                    )
                )
            elif dataset == "reason_seg":
                self.all_datasets.append(
                    ReasonSegDataset(
                        base_image_dir,
                        vision_tower,
                        samples_per_epoch,
                        image_size,
                        num_classes_per_sample,
                        reason_seg_data,
                    )
                )

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        ind = np.random.choice(list(range(len(self.all_datasets))), p=self.sample_rate)
        data = self.all_datasets[ind]
        return data[0]


def collate_fn_val(batch, tokenizer=None, use_mm_start_end=True, padding="right"):
    image_path_list = []
    images_list = []
    images_clip_list = []
    conversation_list = []
    masks_list = []
    exists_list = []
    ref_id_list = []
    sent_id_list = []
    sam_mask_shape_list = []
    offset_list = [0]
    cnt = 0
    conversation_records_list = []
    for (image_path, images, images_clip, conversations,
            masks, sam_mask_shape, exists, ref_id, sent_id, conversation_record) in batch:
        image_path_list.append(image_path)
        images_list.append(images)
        images_clip_list.append(images_clip)
        conversation_list.extend(conversations)
        masks_list.append(masks.float())
        sam_mask_shape_list.append(sam_mask_shape)
        cnt += len(conversations)
        offset_list.append(cnt)
        exists_list.append(exists)
        ref_id_list.append(ref_id)
        sent_id_list.append(sent_id)
        conversation_records_list.append(conversation_record)

    # Replace <image> token if use_mm_start_end is True
    if use_mm_start_end:
        conversation_list = replace_image_tokens(conversation_list)

    # Tokenization and padding of input IDs
    input_ids, attention_masks = tokenize_and_pad(conversation_list, tokenizer, padding=padding)

    return {
        "image_paths": image_path_list,
        "images": torch.stack(images_list, dim=0),
        "images_clip": torch.stack(images_clip_list, dim=0),
        "input_ids": input_ids,
        "labels": None,
        "attention_masks": attention_masks,
        "masks_list": masks_list,
        "sam_mask_shape_list": sam_mask_shape_list,
        "offset": torch.LongTensor(offset_list),
        "inference": True,
        "conversation_list": conversation_list,
        "exists": exists_list,
        "ref_ids": ref_id_list,
        "sent_ids": sent_id_list,
        "conversation_records": conversation_records_list
    }


class TrainValDataset(ReferSegDataset):
    # Use natural referring segmentation dataset as validation set

    def __init__(
        self,
        base_image_dir,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        image_size: int = 336,
        num_classes_per_sample: int = 1,
        train_val_split="val",
        refer_seg_data="refcoco||refcoco+||refcocog",
    ):
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = vision_tower

        self.short_question_list = SHORT_QUESTION_TEMPLATE
        self.answer_list = SHORT_ANSWER_TEMPLATE
        self.neg_answer_list = NEG_ANSWER_TEMPLATE
        self.refer_seg_data = self.load_refer_seg_data(refer_seg_data, train_val_split)

    def __len__(self):
        if len(self.refer_seg_data)>1:
            return self.samples_per_epoch
        else:
            return len(self.refer_seg_data[0]) # for single refer_seg_data

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
            None,                       # ref id (useless now)
            None                        # sent id (useless now)
        )
    
