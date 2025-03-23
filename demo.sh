export CUDA_VISIBLE_DEVICES=0
#python demo.py --pretrained_model_path="../dataset/SESAME-LLaVA-v1.5-7B"
python demo.py \
        --pretrained_model_path="./READ-LLaVA-v1.5-13B-for-ReasonSeg-testset" \
        --vision_tower="../dataset_sesame/clip-vit-large-patch14-336" \
        --model_max_length=2048
