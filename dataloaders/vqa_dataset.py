import json
import os
import random
import torch

from model.llava import conversation as conversation_lib
from model.llava.constants import DEFAULT_IMAGE_TOKEN
from .base_dataset import BaseDataset


def preprocess_multimodal(source):
    for sentence in source:
        if DEFAULT_IMAGE_TOKEN in sentence["value"]:
            sentence["value"] = (
                sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
            )
            sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
            sentence["value"] = sentence["value"].strip()
            if "mmtag" in conversation_lib.default_conversation.version:
                sentence["value"] = sentence["value"].replace(
                    DEFAULT_IMAGE_TOKEN, "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>"
                )
    return source


class VQADataset(BaseDataset):

    def __init__(
        self,
        base_image_dir,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        image_size: int = 336,
        vqa_data="llava_instruct_150k",
    ):
        super().__init__(vision_tower, samples_per_epoch, image_size)
        self.base_image_dir = base_image_dir
        # Load VQA datasets
        DATA_DIR = os.path.join(base_image_dir, "llava_dataset")
        self.vqa_image_root = os.path.join(base_image_dir, "coco/train2017")
        with open(os.path.join(DATA_DIR, "{}.json".format(vqa_data))) as f:
            vqa_data = json.load(f)
        self.vqa_data = vqa_data

        print("vqa_data: ", len(self.vqa_data))

    def __getitem__(self, idx):
        idx = random.randint(0, len(self.vqa_data) - 1)
        item = self.vqa_data[idx]
        image_path = os.path.join(self.vqa_image_root, item["image"])
        # Load images and clip features
        image, image_clip, sam_input_shape = self.load_and_preprocess_image(image_path)
        # Process conversation Q/A
        conv = conversation_lib.default_conversation.copy()
        source = preprocess_multimodal(item["conversations"])
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        conversations = []
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]
        conv.messages = []
        for sentence in source:
            role = roles[sentence["from"]]
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
        # Empty segmentation maps for VQA datasets
        masks = torch.rand(0, *sam_input_shape)
        exists = [False]
        sam_mask_shape = [sam_input_shape, (masks.shape[1], masks.shape[2])]
        return (
            image_path,                 # filename
            image,                      # raw image (for SAM)
            image_clip,                 # image clip feature (for LMMs)
            conversations,              # QA
            masks,                      # segmentation GT
            sam_mask_shape,             # input / output shape for SAM
            exists,                     # object existence
        )