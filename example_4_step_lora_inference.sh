#!/bin/bash


### This script is for 4 step lora inference to speed up inference. 
### Usual time to generate a 1280*720 video is around 3 minuts.
### Set offload to be True if you encounter out of memory issue.


set -e  # Exit on any error

python 4_step_lora_generate.py \
    --task i2v-A14B \
    --size "1280*720" \
    --ckpt_dir ./Models/Wan2.2-I2V-A14B \
    --lora_dir ./Models/Lora/merged_lora \
    --offload_model False \
    --base_seed 42 \
    --prompt_file Data/4_step_lora_examples/i2v_prompt_list.txt \
    --image_path_file Data/4_step_lora_examples/i2v_image_path_list.txt