export CUDA_VISIBLE_DEVICES=0,1,2,3 
python demo.py \
	--pretrained_model_path="../dataset/SESAME-LLaVA-v1.5-7B" \
	--vision_tower="../dataset/clip-vit-large-patch14-336" \
	--model_max_length=2048
