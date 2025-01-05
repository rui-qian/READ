import argparse
import os
import sys
import cv2
import numpy as np
import torch
from model.READ import load_pretrained_model_READ
from model.llava import conversation as conversation_lib
from utils import prepare_input
from model.llava.constants import DEFAULT_IMAGE_TOKEN
from dataloaders.base_dataset import ImageProcessor
from dataloaders.utils import replace_image_tokens, tokenize_and_pad


def parse_args(args):
    parser = argparse.ArgumentParser(description="READ demo")
    parser.add_argument(
                        "--vision_tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--pretrained_model_path", default="ruiqian/READ")
    parser.add_argument("--vis_save_dir", default="./demo_directory", type=str)
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    return parser.parse_args(args)


def save_segmentation(pred_mask, input_dict, args):
    pred_mask = pred_mask.detach().cpu().numpy()
    pred_mask = pred_mask > 0

    image_path = input_dict["image_path"]
    image_key = image_path.split("/")[-1].split(".")[0]
    save_dir = os.path.join(args.vis_save_dir, image_key)
    os.makedirs(save_dir, exist_ok=True)
    seg_fname = os.path.join(save_dir, "seg_mask.jpg")
    cv2.imwrite(seg_fname, pred_mask * 100)
    seg_rgb_fname = os.path.join(save_dir, "seg_rgb.jpg")
    image_np = cv2.imread(image_path)
    image_np[pred_mask] = (
        image_np * 0.3
        + pred_mask[:, :, None].astype(np.uint8) * np.array([0, 0, 255]) * 0.7
    )[pred_mask]
    cv2.imwrite(seg_rgb_fname, image_np)
    return save_dir


@torch.inference_mode()
def demo(args):
    # Initialization
    os.makedirs(args.vis_save_dir, exist_ok=True)

    (
        tokenizer,
        segmentation_lmm,
        vision_tower,
        context_len,
    ) = load_pretrained_model_READ(
        model_path=args.pretrained_model_path,
        vision_tower=args.vision_tower,
        model_max_length=args.model_max_length
    )
    # Load bf16 datatype
    vision_tower = vision_tower.to(torch.bfloat16)
    segmentation_lmm = segmentation_lmm.to(torch.bfloat16)
    segmentation_lmm = torch.compile(segmentation_lmm, mode="reduce-overhead")
    # for eval only
    tokenizer.padding_side = "left"
    
    while True:
        print("---------- User Input ----------")
        print("Press Ctrl-C to exit the demo mode.")
        question = input("Human Prompts: ")
        image_path = input("Image path:")
        # Format images
        img_processor = ImageProcessor(vision_tower.image_processor, args.image_size)
        image, image_clip, sam_mask_shape = img_processor.load_and_preprocess_image(image_path)

        # Format Question
        conv = conversation_lib.default_conversation.copy()
        question = DEFAULT_IMAGE_TOKEN + "\n" + question
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        conversation_list = [conv.get_prompt()]
        mm_use_im_start_end = getattr(segmentation_lmm.config, "mm_use_im_start_end", False)
        if mm_use_im_start_end:
            conversation_list = replace_image_tokens(conversation_list)
        input_ids, _ = tokenize_and_pad(conversation_list, tokenizer, padding='left')
        # Format input dictionary
        input_dict = {
            "image_path": image_path,
            "images_clip": torch.stack([image_clip], dim=0),
            "images": torch.stack([image], dim=0),
            "input_ids": input_ids,
            "sam_mask_shape_list": [sam_mask_shape]
        }
        input_dict = prepare_input(input_dict, "bf16", is_cuda=True)
        output_ids, pred_masks, object_presence = segmentation_lmm.evaluate(
            input_dict["images_clip"],
            input_dict["images"],
            input_dict["input_ids"],
            input_dict["sam_mask_shape_list"],
            max_new_tokens=args.model_max_length,
        )
        real_output_ids = output_ids[:, input_ids.shape[1] :]
        generated_outputs = tokenizer.batch_decode(
            real_output_ids, skip_special_tokens=True
        )[0]
        segmentation_dir = save_segmentation(pred_masks[0], input_dict, args)
        print("---------- Model Output ----------")
        print(f"* Object Existence (See): {object_presence[0]}")
        print(f"* Text Response (Say): {generated_outputs}")
        print(f"* Segmentation Paths (Segment): {segmentation_dir}")


def main(args):
    args = parse_args(args)

    print(args)

    demo(args)
if __name__ == "__main__":
    main(sys.argv[1:])
