#!/bin/bash


set -e  # Exit on any error

root_dir="/home/users/astar/cfar/stuchengyou/jcy/models"

# Use 720x1280 resolution for high resolution (use 640x480 for lower and faster generation)
# This script generates all videos for the test dataset
python ./VideoX-Fun/examples/wan2.2/single_predict.py \
    --resolution 720x1280 \
    --model_name ${root_dir}/Wan2.2-I2V-A14B \
    --lora_low ${root_dir}/FFGO-Lora-Adapter/10_LargeMixedDatset_wan_14bLow_f81_LongCaption_StartMatch_run_r128_a128_3obj_Longrun_B4/checkpoint-600.safetensors \
    --lora_high ${root_dir}/FFGO-Lora-Adapter/10_LargeMixedDatset_wan_14bHigh_f81_LongCaption_StartMatch_run_r128_a128_3obj_Longrun_B4/checkpoint-600.safetensors \
    --config_path ./VideoX-Fun/config/wan2.2/wan_civitai_i2v.yaml \
    --output_path ./output/ffgo_eval \
    --image_path ./Data/combined_first_frames/combined_tryon.png \
    --caption "ad23r2 the camera view suddenly changes. A soft-lit medium shot reveals a young woman with long flowing hair, now wearing a sleek black Nike zip-up hoodie. She gently turns her body, lifting the collar and adjusting the zipper to showcase the fabric texture and clean silhouette. The camera glides around her as she slightly raises her arms, highlighting the fit, pockets, and hood design under clear studio lighting."
