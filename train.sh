#!/bin/bash

export MODEL_NAME="/scratch/prj0000000275/jcy/.cache/huggingface/hub/models--Wan-AI--Wan2.2-I2V-A14B/snapshots/206a9ee1b7bfaaf8f7e4d81335650533490646a3"
export DATASET_NAME="/scratch/prj0000000275/jcy/datasets/tryon_dataset"
export DATASET_META_NAME="/scratch/prj0000000275/jcy/datasets/tryon_dataset/train.json"

accelerate launch --num_processes=8 --multi_gpu --mixed_precision="bf16" \
    VideoX-Fun/scripts/wan2.2/train_full_transition_14b.py \
  --config_path="/home/users/astar/cfar/stuchengyou/FFGO-Video-Customization/VideoX-Fun/config/wan2.2/wan_civitai_i2v.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=832 \
  --video_sample_size=832 \
  --token_sample_size=512 \
  --video_sample_stride=1 \
  --video_sample_n_frames=81 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=4 \
  --dataloader_num_workers=16 \
  --num_train_epochs=100 \
  --checkpointing_steps=100 \
  --learning_rate=1e-05 \
  --seed=42 \
  --output_dir="/scratch/prj0000000275/jcy/tryon/tryon_high_full" \
  --mixed_precision="bf16" \
  --weight_decay=1e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=1.0 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --uniform_sampling \
  --boundary_type="high" \
  --height=832 \
  --width=624 \
  --validation_steps=200000000 \
  --validation_prompts "ad23r2 the camera view suddenly changes. A woman models a coordinated outfit against a classic paneled wall. The ensemble features a white square-neck sleeveless top, a flowing midi skirt with a dark, intricate paisley print, black stiletto sandals adorned with beaded straps, and a structured black handbag with a gold clasp. The video highlights the movement and drape of the skirt as she turns and walks, while the accessories are showcased in static product shots, emphasizing their design and role in completing the look." \
  --validation_images "/scratch/prj0000000275/jcy/datasets/tryon_dataset/images/lower_body_1207997_detail.png" \
  --gradient_checkpointing