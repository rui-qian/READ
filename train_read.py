from model.llava.train.llama_flash_attn_monkey_patch import (
  replace_llama_attn_with_flash_attn,
)
replace_llama_attn_with_flash_attn()

import argparse
import os
import glob
import shutil
import sys
import time
from functools import partial

import deepspeed
import torch
import tqdm

from model.READ import init_READ_model
from model.llava import conversation as conversation_lib

from dataloaders.trainval_dataset import HybridDataset, TrainValDataset, collate_fn_train, collate_fn_val
from dataloaders.test_dataset import TestReasoningDataset, TestReferDataset, collate_fn_test
from torch.utils.tensorboard import SummaryWriter
from utils import (
    AverageMeter,
    ProgressMeter,
    Summary,
    prepare_input,
    intersectionAndUnionGPU,
    random_seed
)

def parse_args(args):
    parser = argparse.ArgumentParser(description="READ Model Training")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument(
        "--version", default="liuhaotian/llava-v1.5-7b"
    )
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision_tower", default="openai/clip-vit-large-patch14-336", type=str
    )
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    parser.add_argument("--dataset", default="refer_seg||correct_refer_seg||vqa||neg_refer_seg", type=str)
    parser.add_argument("--sample_rates", default="9,3,3", type=str)
    parser.add_argument(
        "--sem_seg_data",
        default="ade20k||cocostuff||pascal_part||paco_lvis||mapillary",
        type=str,
    )
    parser.add_argument(
        "--refer_seg_data", default="refclef||refcoco||refcoco+||refcocog", type=str
    )
    parser.add_argument(
        "--neg_refer_seg_data", default="R-refcocog||R-refcoco||R-refcoco+", type=str
    )
    parser.add_argument(
        "--correct_refer_seg_data",
        default="fprefcocog||fprefcoco||fprefcoco+",
        type=str,
    )
    parser.add_argument("--vqa_data", default="llava_instruct_150k", type=str)
    parser.add_argument("--reason_seg_data", default="ReasonSeg|train", type=str)
    parser.add_argument("--val_dataset", default="ReasonSeg", type=str)
    parser.add_argument("--val_split", default="val", type=str)
    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    parser.add_argument("--log_base_dir", default="./runs", type=str)
    parser.add_argument("--exp_name", default="read_referseg", type=str)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--steps_per_epoch", default=500, type=int)
    parser.add_argument(
        "--batch_size", default=12, type=int, help="batch size per device per step"
    )
    parser.add_argument(
        "--grad_accumulation_steps",
        default=1,
        type=int,
    )
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lr", default=0.0003, type=float)    
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--num_classes_per_sample", default=1, type=int)
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--vision_pretrained", default="PATH_TO_SAM_ViT-H", type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--print_freq", default=3, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--auto_resume", action="store_true", default=False)
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--use_released_param", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    random_seed(seed = 42)
    # Create log directory
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    writer = None
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        if args.use_wandb:
            import wandb
            wandb.init(project="read", name=args.exp_name)
        else:
            writer = SummaryWriter(args.log_dir)

    # Init conversation
    conversation_lib.default_conversation = conversation_lib.conv_templates[
        args.conv_type
    ]
    
    # Init model
    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "vision_pretrained": args.vision_pretrained,
        "use_mm_start_end": args.use_mm_start_end,
        "vision_tower": args.vision_tower
    }
    tokenizer, model, vision_tower = init_READ_model(args, model_args)
    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1
    train_dataset = HybridDataset(
        args.dataset_dir,
        vision_tower.image_processor,
        samples_per_epoch=args.batch_size
        * args.grad_accumulation_steps
        * args.steps_per_epoch
        * world_size,
        image_size=args.image_size,
        num_classes_per_sample=args.num_classes_per_sample,
        dataset=args.dataset,
        sample_rate=[float(x) for x in args.sample_rates.split(",")],
        sem_seg_data=args.sem_seg_data,
        refer_seg_data=args.refer_seg_data,
        neg_refer_seg_data=args.neg_refer_seg_data,
        vqa_data=args.vqa_data,
        reason_seg_data=args.reason_seg_data
    )
    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": 0.0,
                "betas": (args.beta1, args.beta2),
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.epochs * args.steps_per_epoch,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": 100,
                "warmup_type": "linear",
            },
        },
        "fp16": {
            "enabled": args.precision == "fp16",
        },
        "bf16": {
            "enabled": args.precision == "bf16",
        },
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        },
    }
    model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
        collate_fn=partial(
            collate_fn_train,
            tokenizer=tokenizer,
            conv_type=args.conv_type,
            use_mm_start_end=args.use_mm_start_end
        ),
        config=ds_config,
    )

    # resume deepspeed checkpoint
    if args.auto_resume and len(args.resume) == 0:
        resume = os.path.join(args.log_dir, "ckpt_model")
        if os.path.exists(resume):
            args.resume = resume
    
    if args.resume:        
        load_path, client_state = model_engine.load_checkpoint(args.resume)
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        args.start_epoch = (
            int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
        )
        print(
            "resume training from {}, start from epoch {}".format(
                args.resume, args.start_epoch
            )
        )
    
    import pickle
    import pdb;pdb.set_trace
    if os.path.exists(os.path.join(args.version, 'conversation_records.pickle')):
        pickle_path = os.path.join(args.version, 'conversation_records.pickle')
    else:
        pickle_path = os.path.join(args.log_dir, 'conversation_records.pickle')
    if args.no_eval == False:
        # HACK: For now, we always use refcoco dataset series for validation
        # val_dataset = TrainValDataset(
        #     args.dataset_dir,
        #     vision_tower.image_processor,
        #     samples_per_epoch=3000, # if len(refer_seg_data)==1, it doesn't work.
        #     image_size=args.image_size,
        #     refer_seg_data="refcoco" # args.refer_seg_data
        # )
        if os.path.exists(pickle_path):
            with open(pickle_path, 'rb') as f:
                conversation_records = pickle.load(f)

        else: 
            conversation_records = {}
        reason_seg_dataset = ["ReasonSeg"]
        refer_seg_dataset = [
            "refcoco", "refcoco+", "refcocog",
        ]
        if args.val_dataset in reason_seg_dataset:
            val_dataset = TestReasoningDataset(
                args.dataset_dir,
                vision_tower.image_processor,
                args.image_size,
                datasetname=args.val_dataset,
                train_test_split=args.val_split, # val|test
                eval_only=args.eval_only,
                conversation_records=conversation_records 
            )
        elif args.val_dataset in refer_seg_dataset:
            val_dataset = TestReferDataset(
                args.dataset_dir,
                vision_tower.image_processor,
                args.image_size,
                datasetname=args.val_dataset,
                train_test_split=args.val_split, # val|test|testA|testB
                conversation_records=conversation_records
        )
        print(
            f"Training with {len(train_dataset)} examples and validating with {len(val_dataset)} examples."
        )
    else:
        val_dataset = None
        print(f"Training with {len(train_dataset)} examples.")

    # validation dataset
    if val_dataset is not None:
        assert args.val_batch_size == 1
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False, drop_last=False
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False,
            sampler=val_sampler,
            collate_fn=partial(
                collate_fn_val,
                tokenizer=tokenizer,
                use_mm_start_end=args.use_mm_start_end
            ),
        )

    train_iter = iter(train_loader)
    best_score, cur_giou = 0.0, 0.0
    if args.resume:
        def find_latest_file(directory, extension):
            files = glob.glob(os.path.join(directory, f'*{extension}'))
            return max(files, key=os.path.getmtime, default=None)
        best_score = float(find_latest_file(args.log_dir, '.pth').split('ciou')[1].split('_')[0])
   
    if args.eval_only:
        giou, ciou, _ = validate(val_loader, model_engine, 0, writer, args)
        exit()

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train_iter, global_iters = train(
            train_loader,
            model_engine,
            epoch,
            scheduler,
            train_iter,
            writer,
            args,
        )

        if args.no_eval == False:
            giou, ciou, conversation_records = validate(val_loader, model_engine, global_iters, writer, args)
            is_best = ciou > best_score
            best_score = max(ciou, best_score)
            cur_giou = giou if is_best else cur_giou

        if args.no_eval or is_best:
            save_dir = os.path.join(args.log_dir, "ckpt_model")
            gather_conversation_records = [None] * world_size 
            torch.distributed.all_gather_object(gather_conversation_records, conversation_records)
            if args.local_rank == 0:
                torch.save(
                    {"epoch": epoch},
                    os.path.join(
                        args.log_dir,
                        "epoch{}_meta_log_ciou{:.3f}_giou{:.3f}.pth".format(
                            epoch, best_score, cur_giou
                        ),
                    ),
                )
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
                merged_conversation_records = {}
                for record in gather_conversation_records:
                    merged_conversation_records.update(record)

                with open(pickle_path, 'wb') as f:
                    conversation_records = pickle.dump(merged_conversation_records, f)
            else:
                merged_conversation_records = {}
            torch.distributed.barrier()
            print(len(merged_conversation_records))
            model_engine.save_checkpoint(save_dir)
            


def train(train_loader, model, epoch, scheduler, train_iter, writer, args):
    """Main training loop."""

    # Initialization of metric trackers
    keys = ["loss", "ce_loss", "mask_bce_loss", "mask_dice_loss", "mask_loss"]
    loss_meters = {}
    for key in keys:
        loss_meters[key] = AverageMeter(key, ":.4f")
    
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    progress = ProgressMeter(
        args.steps_per_epoch,
        [batch_time, data_time]+list(loss_meters.values()),
        prefix="Epoch: [{}]".format(epoch),
    )

    # Train mode
    model.train()
    end = time.time()
    for global_step in range(args.steps_per_epoch):
        for _ in range(args.grad_accumulation_steps):
            try:
                input_dict = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                input_dict = next(train_iter)
            # Prepare inputs and execute model
            input_dict = prepare_input(input_dict, args.precision, is_cuda=True)
            
            data_time.update(time.time() - end)

            output_dict = model(**input_dict)
            # Update loss metrics
            batch_size = input_dict["images"].size(0)
            for key in keys:
                loss_meters[key].update(output_dict[key].item(), batch_size)
            # Backward and optimizer step
            model.backward(output_dict["loss"])
            model.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Logging
        if global_step % args.print_freq == (args.print_freq-1):
            # All-reduce the losses if in a distributed setting
            if args.distributed:
                for key in keys:
                    loss_meters[key].all_reduce()
                batch_time.all_reduce()
                data_time.all_reduce()

            # Logging and resetting losses
            total_steps = global_step + args.steps_per_epoch * epoch
            if args.local_rank == 0:
                progress.display(global_step + 1)
                
                if args.use_wandb:
                    for key in keys:
                        wandb.log({f"{key}": loss_meters[key].avg}, step=total_steps)
                    # Log learning rate
                    curr_lr = scheduler.get_last_lr()[0]
                    wandb.log({"lr": curr_lr}, step=total_steps)
                else:
                    for key in keys:
                        writer.add_scalar(f"{key}", loss_meters[key].avg, total_steps)
                    # Log learning rate
                    curr_lr = scheduler.get_last_lr()[0]
                    writer.add_scalar("lr", curr_lr, total_steps)

        # Reset all the losses
        for key in keys:
            loss_meters[key].reset()
        batch_time.reset()
        data_time.reset()
        
    return train_iter, total_steps


@torch.inference_mode()
def validate(val_loader, model_engine, global_iters, writer, args):
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    model_engine.eval()
    
    conversation_records = {}
    for input_dict in tqdm.tqdm(val_loader):
        torch.cuda.empty_cache()
        input_dict = prepare_input(input_dict, args.precision, is_cuda=True)
        output_dict = model_engine(**input_dict)
        pred_masks = output_dict["pred_masks"]
        masks_list = output_dict["gt_masks"][0].int()
        output_list = (pred_masks[0] > 0).int()
        conversation_records.update(input_dict['conversation_records'][0])
        assert len(pred_masks) == 1

        intersection, union, acc_iou = 0.0, 0.0, 0.0
        for mask_i, output_i in zip(masks_list, output_list):
            intersection_i, union_i, _ = intersectionAndUnionGPU(
                output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
            )
            intersection += intersection_i
            union += union_i
            acc_iou += intersection_i / (union_i + 1e-5)
            acc_iou[union_i == 0] += 1.0  # no-object target
        intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
        acc_iou = acc_iou.cpu().numpy() / masks_list.shape[0]
        intersection_meter.update(intersection), union_meter.update(
            union
        ), acc_iou_meter.update(acc_iou, n=masks_list.shape[0])
        
    # all reduce in distributed setting
    intersection_meter.all_reduce()
    union_meter.all_reduce()
    acc_iou_meter.all_reduce()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]

    if args.local_rank == 0:
        print("giou: {:.4f}, ciou: {:.4f}".format(giou, ciou))
        if args.use_wandb:
            wandb.log({"giou": giou}, step=global_iters)
            wandb.log({"ciou": ciou}, step=global_iters)
        else:
            writer.add_scalar("giou", giou, global_iters)
            writer.add_scalar("ciou", ciou, global_iters)
    return giou, ciou, conversation_records


if __name__ == "__main__":
    main(sys.argv[1:])

