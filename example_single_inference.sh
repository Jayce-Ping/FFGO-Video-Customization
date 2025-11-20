#!/bin/bash


set -e  # Exit on any error


# Use 720x1280 resolution for high resolution (use 640x480 for lower and faster generation)
# This script generates all videos for the test dataset
python ./VideoX-Fun/examples/wan2.2/single_predict.py \
    --resolution 720x1280 \
    --model_name ./Models/Wan2.2-I2V-A14B \
    --lora_low ./Models/Lora/10_LargeMixedDatset_wan_14bLow_f81_LongCaption_StartMatch_run_r128_a128_3obj_Longrun_B4/checkpoint-600.safetensors \
    --lora_high ./Models/Lora/10_LargeMixedDatset_wan_14bHigh_f81_LongCaption_StartMatch_run_r128_a128_3obj_Longrun_B4/checkpoint-600.safetensors \
    --config_path ./VideoX-Fun/config/wan2.2/wan_civitai_i2v.yaml \
    --output_path ./output/ffgo_eval \
    --image_path ./Data/combined_first_frames/31_combined.png \
    --caption "ad23r2 the camera view suddenly changes. The video begins with a wide shot of three young individuals standing side by side in front of an aged stone corridor. Each of them is dressed in dark robes with red-and-gold accents, creating a cohesive visual identity. The young man on the left, with short dark hair and round glasses, raises his left hand to present a sleek black Apple iPhone—its reflective dual cameras glint under natural light, highlighting precision engineering. The woman in the center, her light brown hair tied loosely, holds a mixed reality headset with both hands, turning it slightly to show its curved glass visor and fabric headband as light passes across its surface. The young man on the right, with medium-length hair, places a silver MacBook on the moss-covered stone ledge, opening it partway to reveal its metallic sheen and engraved logo."
