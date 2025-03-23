import glob
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide

from .data_processing import get_mask_from_json

from .qa_template import LONG_QUESTION_TEMPLATE, LONG_ANSWER_TEMPLATE, SHORT_QUESTION_TEMPLATE
from .base_dataset import BaseDataset

class ReasonSegDataset(BaseDataset):
    choices = ["True_Premise",  "False_Premise_Correction"]  
    weights = [0.85, 0.15]
    def __init__(
        self,
        base_image_dir,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        image_size: int = 336,
        num_classes_per_sample: int = 3,
        reason_seg_data="ReasonSeg|train",
    ):
        super().__init__(vision_tower, samples_per_epoch, image_size)
        self.reason_seg_data = reason_seg_data
        self.num_classes_per_sample = num_classes_per_sample
        self.base_image_dir = base_image_dir

        self.short_question_list = SHORT_QUESTION_TEMPLATE
        self.long_question_list = LONG_QUESTION_TEMPLATE
        self.answer_list = LONG_ANSWER_TEMPLATE
        # load dataset
        reason_seg_data, splits = reason_seg_data.split("|")
        splits = splits.split("_")
        images = []
        for split in splits:
            images_split = glob.glob(
                os.path.join(
                    base_image_dir, "reason_seg", reason_seg_data, split, "*.jpg"
                )
            )
            images.extend(images_split)
        jsons = [path.replace(".jpg", ".json") for path in images]
        self.reason_seg_data = (images, jsons)

        print("number of reason_seg samples: ", len(images))

    def __getitem__(self, idx):
        images, jsons = self.reason_seg_data
        idx = random.randint(0, len(images) - 1)
       
        mode_this_turn = random.choices(self.choices, self.weights, k=1)[0]
        image_path = images[idx]
        json_path = jsons[idx]

        # Load images and clip features
        image, image_clip, sam_input_shape = self.load_and_preprocess_image(image_path)
        # Get sents and segmentation maps
        img = cv2.imread(image_path)[:, :, ::-1]
        mask, sents, fp_qa, is_sentence = get_mask_from_json(json_path, img)
        # Sampling
        sample_size = min(len(sents), self.num_classes_per_sample)
        sampled_inds = random.sample(range(len(sents)), sample_size) if len(sents) >= self.num_classes_per_sample else range(len(sents))
        neg_sample_size = min(len(fp_qa), self.num_classes_per_sample)
        neg_sampled_inds = random.sample(range(len(fp_qa)), neg_sample_size) if len(fp_qa) >= self.num_classes_per_sample else range(len(fp_qa))
        
        sampled_sents = [sents[idx] for idx in sampled_inds]
        neg_sampled_sents = [fp_qa[idx] for idx in neg_sampled_inds]
        sampled_masks = [
            (mask == 1).astype(np.float32) for _ in range(len(sampled_inds))
        ]
        # Create Q/A Data
        questions = []
        answers = []
        conversations = []
        for idx, text in enumerate(sampled_sents):
            neg_text = neg_sampled_sents[idx]
            if mode_this_turn == "True_Premise":
                if is_sentence:
                    question_template = random.choice(self.long_question_list)
                    questions.append(question_template.format(sent=text))
                else:
                    question_template = random.choice(self.short_question_list)
                    questions.append(question_template.format(class_name=text.lower()))
                answers.append(random.choice(self.answer_list))
            else:
                if neg_text[1] is True:
                    question_template = random.choice(self.long_question_list)
                    questions.append(question_template.format(sent=neg_text[0]))
                else:
                    question_template = random.choice(self.short_question_list)
                    questions.append(question_template.format(class_name=neg_text[0]))
                answers.append(neg_text[2])
                
            conv = conversation_lib.default_conversation.copy()
            conv.append_message(conv.roles[0], questions[idx])
            conv.append_message(conv.roles[1], answers[idx])
            conversations.append(conv.get_prompt())

        if mode_this_turn != "True_Premise":
            sampled_sents = neg_sampled_sents
            exists = [False for _ in range(len(sampled_sents))]
            masks = torch.rand(0, *sam_input_shape)
        else:
            exists = [True for _ in range(len(sampled_sents))]
            masks = np.stack(sampled_masks, axis=0)
            masks = torch.from_numpy(masks)
        sam_mask_shape = [sam_input_shape, (masks.shape[1], masks.shape[2])]
        return (
            image_path,         # filename
            image,              # raw image (for SAM)
            image_clip,         # image clip feature (for LMMs)
            conversations,      # QA
            masks,              # segmentation GT
            sam_mask_shape,     # input / output shape for SAM
            exists,             # object existence
        )
