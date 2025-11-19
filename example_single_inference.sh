#!/bin/bash


set -e  # Exit on any error


python ./VideoX-Fun/examples/wan2.2/single_predict.py \
    --resolution 720x1280 \
    --model_name ./Models/Wan2.2-I2V-A14B \
    --lora_low ./Models/Lora/10_LargeMixedDatset_wan_14bLow_f81_LongCaption_StartMatch_run_r128_a128_3obj_Longrun_B4/checkpoint-600.safetensors \
    --lora_high ./Models/Lora/10_LargeMixedDatset_wan_14bHigh_f81_LongCaption_StartMatch_run_r128_a128_3obj_Longrun_B4/checkpoint-600.safetensors \
    --config_path ./VideoX-Fun/config/wan2.2/wan_civitai_i2v.yaml \
    --output_path ./output/ffgo_eval \
    --image_path ./Data/fun2_8.png \
    --caption "Film quality, high resolution, photorealistic. In a sunlit, elegant room with white wooden floorboards and walls adorned with cascading arrangements of pink and red roses, a stylish young asian man Cai Xukun stands centrally. Cai Xukun is dressed in a black turtleneck, beige suspenders, and grey plaid trousers, with a playful expression on his face. On the tip of his right index finger, he expertly balances a spinning orange basketball. In his left hand, he securely holds a white plate piled high with crispy, golden-brown fried chicken. Down by his feet on the wooden floor, a live reddish-brown hen with a bright red comb struts energetically next to him, creating a dynamic contrast between the sports element, the food, and the live animal."