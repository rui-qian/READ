import argparse
import os
import sys
import json
from functools import partial
import tqdm
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from dataloaders.test_dataset import TestReasoningDataset, TestReferDataset, collate_fn_test
from model.READ import load_pretrained_model_READ
from utils import (
    prepare_input,
    intersectionAndUnionGPU,
    AverageMeter,
    Summary,
    random_seed
)

def parse_args(args):
    parser = argparse.ArgumentParser(description="READ's evaluation helper")
    parser.add_argument("--cmd", default="inference", type=str, help="Value can be: 'inference' (default) or 'metrics' 'download'.")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--process_num", default=0, type=int, help="For splitting inference across multiple processes/GPUs. The value should be between 0 and world_size-1.")
    parser.add_argument("--world_size", default=2, type=int, help="total node count")
    parser.add_argument("--pretrained_model_path", default="ruiqian/READ")
    parser.add_argument("--vis_save_path", default="./save_directory", type=str)
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--val_dataset", default="fprefcoco", type=str)
    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    parser.add_argument(
                        "--vision_tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    return parser.parse_args(args)


def save_preds(val_dataset, preds, process_num, world_size, inference_dir):
    filename = f"preds_{val_dataset}_{process_num}_of_{world_size}.json"
    file_path = os.path.join(inference_dir, filename)
    with open(file_path, "w") as file:
        json.dump(preds, file)


def load_preds_file(val_dataset, process_num, world_size, inference_dir):
    filename = f"preds_{val_dataset}_{process_num}_of_{world_size}.json"
    file_path = os.path.join(inference_dir, filename)
    with open(file_path, "rb") as file:
        return json.load(file)

@torch.inference_mode()
def inference(args):
    # Initialization
    os.makedirs(args.vis_save_path, exist_ok=True)
    os.makedirs(os.path.join(args.vis_save_path, "segmentation_mask"), exist_ok=True)

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
    reason_seg_dataset = ["ReasonSeg"]
    refer_seg_dataset = [
        "fprefcoco", "fprefcoco+", "fprefcocog",
        "refcoco", "refcoco+", "refcocog",
        "R-refcoco", "R-refcoco+", "R-refcocog"
    ]

    if args.val_dataset in reason_seg_dataset:
        test_dataset = TestReasoningDataset(
            args.dataset_dir,
            vision_tower.image_processor,
            args.image_size,
            datasetname=args.val_dataset,
            train_test_split="val",
            use_val_mode=False,
            use_test_mode=True
        )
    elif args.val_dataset in refer_seg_dataset:
        test_dataset = TestReferDataset(
            args.dataset_dir,
            vision_tower.image_processor,
            args.image_size,
            datasetname=args.val_dataset,
            train_test_split="val",
            use_val_mode=False,
            use_test_mode=True
        )
    test_dataset = get_dataset_slice(test_dataset, args.process_num, args.world_size, debug=False)
    test_loader = DataLoader(
        test_dataset, batch_size=1, num_workers=1,
        shuffle=False, drop_last=False, pin_memory=False,
        collate_fn=partial(
            collate_fn_test,
            tokenizer=tokenizer,
            use_mm_start_end=getattr(segmentation_lmm.config, "mm_use_im_start_end", False),
            padding="left",
        ),
    )
    idx = 0
    output_json = {}

    # Process each item in the test loader
    for input_dict in tqdm.tqdm(test_loader):
        idx += 1
        input_dict = prepare_input(input_dict, "bf16", is_cuda=True)
        N = len(input_dict["input_ids"])
        # total_count += N
        assert N == len(input_dict["exists"][0])

        # Batch processing setup
        batch_size = 1
        num_batch = math.ceil(N / batch_size)
        pred_masks = []
        pred_exists = []
        gt_masks = input_dict["masks_list"][0].int()
        
        # Prepare output JSON structure
        image_file_key = input_dict["image_paths"][0]
        output_json[image_file_key] = {}

        raw_questions = [
            x.split("\n")[1].split("ASSISTANT:")[0].strip()
            for x in input_dict["conversation_list"]
        ]
        output_json[image_file_key] = {
            "conversation_list": raw_questions,
            "pred_sent": [],
            "gt_exists": input_dict["exists"][0],
            "ref_ids": input_dict["ref_ids"],
            "sent_ids": input_dict["sent_ids"]
        }

        # Batch inference to prevent OOM
        for n in range(num_batch):
            start_idx = n * batch_size
            end_idx = min((n + 1) * batch_size, N)
            input_ids = input_dict["input_ids"][start_idx:end_idx]
            real_batch_size = input_ids.shape[0]

            # Image preparation
            images_clip = input_dict["images_clip"].repeat(
                real_batch_size, 1, 1, 1
            )  # N x 3 x 336 x 336
            images = input_dict["images"].repeat(
                real_batch_size, 1, 1, 1
            )  # N x 3 x 1024 x 1024
            sam_mask_shape_list = input_dict["sam_mask_shape_list"] * real_batch_size
            # Model evaluation
            with torch.inference_mode():
                output_ids, pred_masks_batch, object_presence = segmentation_lmm.evaluate(
                    images_clip,
                    images,
                    input_ids,
                    sam_mask_shape_list,
                    max_new_tokens=512,
                )
                pred_exists += object_presence
                pred_masks += pred_masks_batch
                real_output_ids = output_ids[:, input_ids.shape[1] :]
                generated_outputs = tokenizer.batch_decode(
                    real_output_ids, skip_special_tokens=True
                )
                output_json[image_file_key]["pred_sent"] += generated_outputs
        # Save results
        pred_masks = torch.stack(pred_masks, dim=0)
        output_json[image_file_key]["pred_exists"] = pred_exists
        output_json[image_file_key]["segmentation_path"] = os.path.join(
            "segmentation_mask", f"{args.process_num}_{idx:04d}.npz"
        )
        output_seg_fname = os.path.join(
            args.vis_save_path,
            output_json[image_file_key]["segmentation_path"],
        )
        np.savez_compressed(
            output_seg_fname,
            pred=pred_masks.cpu().numpy(),
            gt=gt_masks.cpu().numpy(),
        )
    save_preds(args.val_dataset, output_json, args.process_num, args.world_size, args.vis_save_path)


def get_dataset_slice(
    val_dataset, process_num: int, world_size: int, debug: bool = False
) -> Subset:
    """
    Return a torch.utils.data.Subset object that is a subset of `val_dataset`.
    `val_dataset` is broken into roughly equal chunks. Number of chunks = `world_size`. The Nth
    chunk is returned (where N = `process_num`).
    """
    all_indices = np.array(range(len(val_dataset)))
    print(f"Splitting Total images: {len(all_indices)}, world_size: {world_size}")
    splits = np.array_split(all_indices, world_size)
    print("Total splits generated: ", len(splits))
    print(
        "Split sizes (each size is for one process/process_num): ",
        [len(s) for s in splits],
    )
    subset_indices = splits[process_num]
    if debug:
        # Force the dataset size to be tiny so we can test the script
        subset_indices = subset_indices[:10]
    print("Indices for current process: ", len(subset_indices))
    subset = Subset(val_dataset, indices=subset_indices)
    print("Subset dataset size: ", len(subset))
    return subset


def run_eval(preds, args):
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    total_count = 0
    correct_query = 0
    for idx, pred_key in enumerate(preds):
        pred = preds[pred_key]
        # Evaluate SEE (binary classification)
        correct_this_round = np.sum(
            np.array(pred["gt_exists"]) == np.array(pred["pred_exists"])
        )
        total_count += np.array(pred["gt_exists"]).shape[0]
        correct_query += correct_this_round
        # Evaluate Say (Pass)

        # Evaluate Segment (IoU)
        seg_fname = os.path.join(
            args.vis_save_path,
            pred["segmentation_path"],
        )
        segmentation = np.load(seg_fname)
        pred_masks = torch.from_numpy(segmentation["pred"]).cuda()
        gt_masks =torch.from_numpy(segmentation["gt"]).cuda()

        intersection, union, _ = intersectionAndUnionGPU(
            pred_masks.contiguous(), gt_masks.contiguous(), 2, ignore_index=255,
        )
        intersection_meter.update(intersection), union_meter.update(union)
        if idx % 50 == 49:
            iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
            detection_acc = correct_query / total_count
            print(
                f"[(batch {idx}) - {args.val_dataset}] cIoU: {iou_class[1]:.4f} | See Accuracy: {detection_acc:.4f}"
            )
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    detection_acc = correct_query / total_count
    print(
        f"[Final Result - {args.val_dataset}] cIoU: {iou_class[1]:.4f} | See Accuracy: {detection_acc:.4f}"
    ) 

import numpy as np
import cv2
from  matplotlib import pyplot as plt
def visualize():
    with open('inference_results/fprefcoco_inference_cvpr/preds_fprefcoco_0_of_1.json', "rb") as f:
        preds = json.load(f)
    # import pdb;pdb.set_trace()
    for idx, pred_key in enumerate(preds):
        # import pdb;pdb.set_trace()
        pred = preds[pred_key]
        seg_fname = os.path.join(
            'inference_results/fprefcoco_inference_cvpr',
            pred["segmentation_path"],
        )
        segmentation = np.load(seg_fname)
        pred_masks = segmentation["pred"]
        gt_masks = segmentation["gt"]
        cv2_img = cv2.imread(pred_key)[:, :, ::-1]
        for i in range(pred_masks.shape[0]):
            vis = cv2_img.copy()
            vis[gt_masks[i] > 0] = vis[gt_masks[i] > 0] // 2 + np.array([0, 255, 0], dtype=np.uint8) // 2
            save_path = 'inference_results/fprefcoco_inference_cvpr/' + pred_key.split('/')[-1].replace('.jpg','_')+str(i)+'.jpg'
            print(save_path)
            plt.imsave(save_path, vis, cmap='viridis')
            # plt.imshow(vis)
            # plt.show()
            vis = cv2_img.copy()
            vis[pred_masks[i] > 0] = vis[pred_masks[i] > 0] // 2 + np.array([255, 0, 0], dtype=np.uint8) // 2
            save_path = 'inference_results/fprefcoco_inference_cvpr/' + pred_key.split('/')[-1].replace('.jpg','_')+str(i)+'.jpg'
            print(save_path)
            plt.imsave(save_path, vis, cmap='viridis')
            # plt.imshow(vis)
            # plt.show()
        # import pdb;pdb.set_trace()

def main(args):
    args = parse_args(args)
    random_seed(seed = 42)
    print(args)
    # visualize()  #uncomment it to visualize test mode results.

    if args.cmd == "inference":
        inference(args)
    elif args.cmd == "metrics":
        combined_preds = {}
        for rank in range(args.world_size):
            combined_preds |= load_preds_file(args.val_dataset, rank, args.world_size, args.vis_save_path)
        run_eval(combined_preds, args)

if __name__ == "__main__":
    main(sys.argv[1:])
