#!/bin/bash


set -e  # Exit on any error

# Use 720x1080 resolution for high resolution (use 480x640 for lower and faster generation)
# This script generates all videos for the test dataset
python ./VideoX-Fun/examples/wan2.2/batch_predict-v1.py \
    --resolution 720x1080 \
    --model_name ./Models/Wan2.2-I2V-A14B \
    --lora_low ./Models/Lora/checkpoint-600-low.safetensors \
    --lora_high ./Models/Lora/checkpoint-600-high.safetensors \
    --config_path ./VideoX-Fun/config/wan2.2/wan_civitai_i2v.yaml \
    --data_csv ./Data/combined_first_frames/0-data.csv \
    --output_path ./output/ffgo_eval