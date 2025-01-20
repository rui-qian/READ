# # uncomment to eval for ReasonSeg val set
#deepspeed --master_port=24996 --include "localhost:0" train_read.py \
#       --version="./READ-LLaVA-v1.5-7B-for-ReasonSeg-valset" \
#       --dataset_dir='../dataset' \
#       --vision_pretrained="../dataset/sam_vit_h_4b8939.pth" \
#       --eval_only \
#       --vision_tower="../dataset/clip-vit-large-patch14-336" \
#       --model_max_length=2048 \
#       --val_dataset="ReasonSeg" \
#       --val_split="val" 

# uncomment to eval for ReasonSeg test set
#deepspeed --master_port=24996 --include "localhost:0" train_read.py \
#       --version="./READ-LLaVA-v1.5-7B-for-ReasonSeg-testset" \
#       --dataset_dir='../dataset' \
#       --vision_pretrained="../dataset/sam_vit_h_4b8939.pth" \
#       --eval_only \
#       --vision_tower="../dataset/clip-vit-large-patch14-336" \
#       --model_max_length=2048 \
#       --val_dataset="ReasonSeg" \
#       --val_split="test"

# uncomment to eval for ReasonSeg test set with 13B model
deepspeed --master_port=24996 --include "localhost:0" train_read.py \
       --version="./READ-LLaVA-v1.5-13B-for-ReasonSeg-testset" \
       --dataset_dir='../dataset' \
       --vision_pretrained="../dataset/sam_vit_h_4b8939.pth" \
       --eval_only \
       --vision_tower="../dataset/clip-vit-large-patch14-336" \
       --model_max_length=2048 \
       --val_dataset="ReasonSeg" \
       --val_split="test"

# # uncomment to eval for refcoco series "refcoco", "refcoco+", "refcocog"
# deepspeed --master_port=24996 --include "localhost:0" train_read.py \
#        --version="./READ-LLaVA-v1.5-7B-for-fprefcoco" \
#        --dataset_dir='../dataset' \
#        --vision_pretrained="../dataset/sam_vit_h_4b8939.pth" \
#        --eval_only \
#        --vision_tower="../dataset/clip-vit-large-patch14-336" \
#        --model_max_length=2048 \
#        --val_dataset="refcoco" \
#        --val_split="val"  # val|test|testA|testB


