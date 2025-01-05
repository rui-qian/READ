#!/bin/bash
function run_inference() {
    CUDA_DEVICE="${1}"
    PROCESS_NUM="${2}"
    WORLD_SIZE="${3}"
    DATASET="${4}"
    INFERENCE_CMD="${5:-inference}"
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" python test_read.py \
        --cmd="${INFERENCE_CMD}" \
        --local_rank=0 \
        --process_num="${PROCESS_NUM}" \
        --world_size="${WORLD_SIZE}" \
        --dataset_dir ../dataset \
        --pretrained_model_path="./READ-LLaVA-v1.5-7B-for-fprefcoco" \
        --vision_tower="../dataset/clip-vit-large-patch14-336" \
        --model_max_length=2048 \
        --val_dataset="${DATASET}" \
        --vis_save_path="./inference_results/${DATASET}_inference_cvpr"
}

# Example: Run inference on CUDA devices 0-7 in parallel:

#declare -a datasets=("refcoco" "refcoco+" "refcocog" "fprefcoco" "fprefcoco+" "fprefcocog")
declare -a datasets=("fprefcoco" "fprefcoco+" "fprefcocog")
# run_inference 0 0 1 "fprefcocog"
for dataset in "${datasets[@]}"; do
    echo "Running inference for ${dataset}..."
    run_inference 0 0 1 "${dataset}"
    echo "Waiting for background inference processes to finish... for ${dataset}..."
    wait
    echo "Background processes for ${dataset} finished. Running metrics..."
    run_inference 0 0 1 "${dataset}" "metrics"
    echo "Inference and metrics for ${dataset} finished."
done

# declare -a datasets=("ReasonSeg")
# for dataset in "${datasets[@]}"; do
#     echo "Running inference for ${dataset}..."
#     run_inference 0 0 8 "${dataset}" &
#     run_inference 1 1 8 "${dataset}" &
#     run_inference 2 2 8 "${dataset}" &
#     run_inference 3 3 8 "${dataset}" &
#     run_inference 4 4 8 "${dataset}" &
#     run_inference 5 5 8 "${dataset}" &
#     run_inference 6 6 8 "${dataset}" &
#     run_inference 7 7 8 "${dataset}" &
#     echo "Waiting for background inference processes to finish... for ${dataset}..."
#     wait
#     echo "Background processes for ${dataset} finished. Running metrics..."
#     run_inference 0 0 8 "${dataset}" "metrics"
#     echo "Inference and metrics for ${dataset} finished."
# done

