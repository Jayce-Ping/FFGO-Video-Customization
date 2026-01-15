#!/bin/bash


set -e  # Exit on any error

# Use 720x1280 resolution for high resolution (use 480x640 for lower and faster generation)
# This script generates all videos for the test dataset
root_dir="/home/users/astar/cfar/stuchengyou/jcy/"

NUM_GPUS=2

resolution="640x480"

python VideoX-Fun/examples/wan2.2/batch_predict-dist.py \
	--resolution ${resolution} \
    --model_name ${root_dir}/models/Wan2.2-I2V-A14B \
    --lora_low ${root_dir}/tryon/tryon_low/checkpoint-600.safetensors \
    --lora_high ${root_dir}/tryon/tryon_high/checkpoint-600.safetensors \
    --config_path ./VideoX-Fun/config/wan2.2/wan_civitai_i2v.yaml \
    --dataset_dir ${root_dir}/datasets/tryon_dataset \
    --data_csv ${root_dir}/datasets/tryon_dataset/test.csv \
    --output_path ${root_dir}/tryon/infer_res/${resolution} \
    --num_gpus ${NUM_GPUS}
