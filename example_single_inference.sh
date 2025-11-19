#!/bin/bash


set -e  # Exit on any error

# Use 720x1280 resolution for high resolution (use 480x640 for lower and faster generation)
# This script generates all videos for the test dataset
python ./VideoX-Fun/examples/wan2.2/single_predict.py \
    --resolution 720x1280 \
    --model_name ./Models/Wan2.2-I2V-A14B \
    --lora_low ./Models/Lora/10_LargeMixedDatset_wan_14bLow_f81_LongCaption_StartMatch_run_r128_a128_3obj_Longrun_B4/checkpoint-600.safetensors \
    --lora_high ./Models/Lora/10_LargeMixedDatset_wan_14bHigh_f81_LongCaption_StartMatch_run_r128_a128_3obj_Longrun_B4/checkpoint-600.safetensors \
    --config_path ./VideoX-Fun/config/wan2.2/wan_civitai_i2v.yaml \
    --output_path ./output/ffgo_eval \
    --image_path ./Data/fun2_7.png \
    --caption "Film quality, high resolution, realistic texture. The video features a young man with brown hair, dressed in a pink sweater, ripped blue jeans, and white sneakers, standing in a bright, rustic room filled with climbing pink and red roses and sunlight streaming through large windows. He is holding a white plate piled high with crispy, golden-brown fried chicken, positioning it centrally in his hands. To his side, a live brown hen with reddish feathers stands on the white wooden floor, strutting slightly. The camera captures a full-body view, highlighting the interaction between the man holding the savory dish and the live animal beside him against the soft, romantic floral background."