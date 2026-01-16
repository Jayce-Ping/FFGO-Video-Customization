"""
I2V Full Fine-tuning Script with Complete Dataset
使用完整数据集功能的 image-to-video 全参数微调
"""

import argparse
import gc
import logging
import math
import os
import sys
import pickle
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, ProjectConfiguration
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_loss_weighting_for_sd3
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, RandomSampler
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from einops import rearrange
import numpy as np

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

# 导入完整的数据集
from videox_fun.data.dataset_transition import (
    ImageVideoDataset,
    ImageVideoSampler,
    get_random_mask
)
from videox_fun.data.bucket_sampler import (
    ASPECT_RATIO_512,
    ASPECT_RATIO_RANDOM_CROP_512,
    ASPECT_RATIO_RANDOM_CROP_PROB,
    AspectRatioBatchImageVideoSampler,
    get_closest_ratio
)
from videox_fun.models import (
    AutoencoderKLWan,
    AutoencoderKLWan3_8,
    WanT5EncoderModel,
    Wan2_2Transformer3DModel
)
from videox_fun.pipeline import Wan2_2I2VPipeline
from videox_fun.utils.utils import get_image_to_video_latent, save_videos_grid
from videox_fun.utils.discrete_sampler import DiscreteSampling

logger = get_logger(__name__)


def filter_kwargs(cls, kwargs):
    """过滤无效参数"""
    import inspect
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self', 'cls'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return filtered_kwargs


def get_random_downsample_ratio(sample_size, image_ratio=[], all_choices=False, rng=None):
    """获取随机下采样比例"""
    def _create_special_list(length):
        if length == 1:
            return [1.0]
        if length >= 2:
            first_element = 0.75
            remaining_sum = 1.0 - first_element
            other_elements_value = remaining_sum / (length - 1)
            special_list = [first_element] + [other_elements_value] * (length - 1)
            return special_list
    
    if sample_size >= 1536:
        number_list = [1, 1.25, 1.5, 2, 2.5, 3] + image_ratio
    elif sample_size >= 1024:
        number_list = [1, 1.25, 1.5, 2] + image_ratio
    elif sample_size >= 768:
        number_list = [1, 1.25, 1.5] + image_ratio
    elif sample_size >= 512:
        number_list = [1] + image_ratio
    else:
        number_list = [1]
    
    if all_choices:
        return number_list
    
    number_list_prob = np.array(_create_special_list(len(number_list)))
    if rng is None:
        return np.random.choice(number_list, p=number_list_prob)
    else:
        return rng.choice(number_list, p=number_list_prob)


def resize_mask(mask, latent, process_first_frame_only=True):
    """调整遮罩大小"""
    latent_size = latent.size()
    batch_size, channels, num_frames, height, width = mask.shape
    
    if process_first_frame_only:
        target_size = list(latent_size[2:])
        target_size[0] = 1
        first_frame_resized = F.interpolate(
            mask[:, :, 0:1, :, :],
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
        
        target_size = list(latent_size[2:])
        target_size[0] = target_size[0] - 1
        if target_size[0] != 0:
            remaining_frames_resized = F.interpolate(
                mask[:, :, 1:, :, :],
                size=target_size,
                mode='trilinear',
                align_corners=False
            )
            resized_mask = torch.cat([first_frame_resized, remaining_frames_resized], dim=2)
        else:
            resized_mask = first_frame_resized
    else:
        target_size = list(latent_size[2:])
        resized_mask = F.interpolate(
            mask,
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
    return resized_mask


def log_validation(vae, text_encoder, tokenizer, transformer, config, args, 
                   accelerator, weight_dtype, global_step, h, w):
    """验证函数"""
    try:
        logger.info("Running validation...")
        
        # 创建验证用的 transformer
        transformer_val = Wan2_2Transformer3DModel.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, f'{args.boundary_type}_noise_model'),
            transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        ).to(weight_dtype)
        transformer_val.load_state_dict(accelerator.unwrap_model(transformer).state_dict())
        
        # 创建调度器
        scheduler = FlowMatchEulerDiscreteScheduler(
            **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
        )
        
        # 创建 pipeline
        pipeline = Wan2_2I2VPipeline(
            vae=accelerator.unwrap_model(vae).to(weight_dtype),
            text_encoder=accelerator.unwrap_model(text_encoder),
            tokenizer=tokenizer,
            transformer=transformer_val,
            scheduler=scheduler,
        )
        pipeline = pipeline.to(accelerator.device)
        
        # 生成验证样本
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
        
        for i in range(len(args.validation_prompts)):
            with torch.no_grad():
                with torch.autocast("cuda", dtype=weight_dtype):
                    video_length = args.video_sample_n_frames
                    input_video, input_video_mask, _ = get_image_to_video_latent(
                        args.validation_images[i], None, 
                        video_length=video_length, 
                        sample_size=[h, w]
                    )
                    
                    sample = pipeline(
                        args.validation_prompts[i],
                        num_frames=video_length,
                        negative_prompt="bad detailed",
                        height=h,
                        width=w,
                        guidance_scale=6.0,
                        generator=generator,
                        video=input_video,
                        mask_video=input_video_mask,
                    ).videos
                    
                    os.makedirs(os.path.join(args.output_dir, "sample"), exist_ok=True)
                    save_videos_grid(sample, os.path.join(args.output_dir, f"sample/sample-{global_step}-{i}.mp4"))
        
        del pipeline
        del transformer_val
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
    except Exception as e:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print(f"Validation error: {e}")
        return None


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="I2V Full Fine-tuning")
    
    # ==================== 模型和数据 ====================
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True, help="预训练模型路径")
    parser.add_argument("--config_path", type=str, required=True, help="模型配置文件")
    parser.add_argument("--train_data_dir", type=str, required=True, help="训练数据目录")
    parser.add_argument("--train_data_meta", type=str, required=True, help="训练数据元文件")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="输出目录")
    parser.add_argument("--cache_dir", type=str, default=None, help="缓存目录")
    
    # ==================== 训练参数 ====================
    parser.add_argument("--train_batch_size", type=int, default=1, help="训练批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--num_train_epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--max_train_steps", type=int, default=None, help="最大训练步数")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="学习率")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", help="学习率调度器")
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="预热步数")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="权重衰减")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="最大梯度范数")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Adam beta2")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="Adam epsilon")
    
    # ==================== 视频参数 ====================
    parser.add_argument("--video_sample_size", type=int, default=832, help="视频采样尺寸")
    parser.add_argument("--image_sample_size", type=int, default=832, help="图像采样尺寸")
    parser.add_argument("--token_sample_size", type=int, default=512, help="token采样尺寸")
    parser.add_argument("--video_sample_stride", type=int, default=1, help="视频采样步长")
    parser.add_argument("--video_sample_n_frames", type=int, default=81, help="视频采样帧数")
    parser.add_argument("--video_repeat", type=int, default=1, help="视频重复次数")
    parser.add_argument("--height", type=int, default=832, help="高度")
    parser.add_argument("--width", type=int, default=624, help="宽度")
    parser.add_argument("--fix_sample_size", nargs=2, type=int, default=None, help="固定采样尺寸")
    
    # ==================== 数据增强 ====================
    parser.add_argument("--random_flip", action="store_true", help="随机翻转")
    parser.add_argument("--random_hw_adapt", action="store_true", help="随机高宽适配")
    parser.add_argument("--random_ratio_crop", action="store_true", help="随机比例裁剪")
    parser.add_argument("--random_frame_crop", action="store_true", help="随机帧裁剪")
    parser.add_argument("--enable_bucket", action="store_true", help="启用bucket采样")
    parser.add_argument("--training_with_video_token_length", action="store_true", help="使用视频token长度训练")
    
    # ==================== 其他设置 ====================
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--dataloader_num_workers", type=int, default=16, help="数据加载器工作线程数")
    parser.add_argument("--vae_mini_batch", type=int, default=1, help="VAE mini batch大小")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="梯度检查点")
    parser.add_argument("--checkpointing_steps", type=int, default=500, help="检查点保存步数")
    parser.add_argument("--checkpoints_total_limit", type=int, default=None, help="最大检查点数量")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="从检查点恢复")
    parser.add_argument("--logging_steps", type=int, default=10, help="日志记录步数")
    
    # ==================== Flow Matching 参数 ====================
    parser.add_argument("--boundary_type", type=str, default="high", choices=["low", "high", "full"])
    parser.add_argument("--train_sampling_steps", type=int, default=1000, help="训练采样步数")
    parser.add_argument("--uniform_sampling", action="store_true", help="均匀采样")
    parser.add_argument("--weighting_scheme", type=str, default="none", 
                       choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"])
    parser.add_argument("--logit_mean", type=float, default=0.0, help="logit均值")
    parser.add_argument("--logit_std", type=float, default=1.0, help="logit标准差")
    parser.add_argument("--mode_scale", type=float, default=1.29, help="mode scale")
    
    # ==================== 验证参数 ====================
    parser.add_argument("--validation_prompts", type=str, nargs="+", default=None, help="验证提示词")
    parser.add_argument("--validation_images", type=str, nargs="+", default=None, help="验证图像")
    parser.add_argument("--validation_steps", type=int, default=200000000, help="验证步数")
    
    # ==================== Tokenizer 参数 ====================
    parser.add_argument("--tokenizer_max_length", type=int, default=512, help="tokenizer最大长度")
    parser.add_argument("--enable_text_encoder_in_dataloader", action="store_true", 
                       help="在dataloader中启用文本编码器")
    
    # ==================== 其他训练技巧 ====================
    parser.add_argument("--multi_stream", action="store_true", help="多CUDA流")
    parser.add_argument("--noise_offset", type=float, default=0, help="噪声偏移")
    parser.add_argument("--input_perturbation", type=float, default=0, help="输入扰动")
    parser.add_argument("--auto_tile_batch_size", action="store_true", help="自动平铺batch大小")
    parser.add_argument("--keep_all_node_same_token_length", action="store_true", 
                       help="保持所有节点相同token长度")
    
    args = parser.parse_args()
    
    # 处理本地rank
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1:
        args.local_rank = env_local_rank
    
    return args


def main():
    args = parse_args()
    
    # ====================================
    # 1. 初始化 Accelerator
    # ====================================
    logging_dir = os.path.join(args.output_dir, "logs")
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=logging_dir
    )
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_config=accelerator_project_config,
    )
    
    # 设置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    # 设置随机种子
    if args.seed is not None:
        set_seed(args.seed)
        rng = np.random.default_rng(np.random.PCG64(args.seed + accelerator.process_index))
        torch_rng = torch.Generator(accelerator.device).manual_seed(args.seed + accelerator.process_index)
    else:
        rng = None
        torch_rng = None
    
    # 创建输出目录
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # 权重数据类型
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    # ====================================
    # 2. 加载模型
    # ====================================
    logger.info("加载模型...")
    config = OmegaConf.load(args.config_path)
    
    # Scheduler
    noise_scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
    )
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, 
                    config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer'))
    )
    
    # Text Encoder（冻结）
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path,
                    config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
        additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )
    text_encoder.eval()
    text_encoder.requires_grad_(False)
    
    # VAE（冻结）
    Chosen_AutoencoderKL = {
        "AutoencoderKLWan": AutoencoderKLWan,
        "AutoencoderKLWan3_8": AutoencoderKLWan3_8
    }[config['vae_kwargs'].get('vae_type', 'AutoencoderKLWan')]
    
    vae = Chosen_AutoencoderKL.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path,
                    config['vae_kwargs'].get('vae_subpath', 'vae')),
        additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
    )
    vae.eval()
    vae.requires_grad_(False)
    
    # Transformer（全参训练）
    if args.boundary_type == "low" or args.boundary_type == "full":
        sub_path = config['transformer_additional_kwargs'].get('transformer_low_noise_model_subpath', 'transformer')
    else:
        sub_path = config['transformer_additional_kwargs'].get('transformer_high_noise_model_subpath', 'transformer')
    
    transformer = Wan2_2Transformer3DModel.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, sub_path),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
    ).to(weight_dtype)
    
    # **关键：启用全参数训练**
    transformer.requires_grad_(True)
    
    # 启用梯度检查点
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    
    logger.info(f"Transformer 参数总数: {sum(p.numel() for p in transformer.parameters()):,}")
    logger.info(f"可训练参数数: {sum(p.numel() for p in transformer.parameters() if p.requires_grad):,}")
    
    # ====================================
    # 3. 准备数据集（使用完整的 ImageVideoDataset）
    # ====================================
    logger.info("准备数据集...")
    
    # 计算 sample_n_frames_bucket_interval
    sample_n_frames_bucket_interval = vae.config.temporal_compression_ratio
    spatial_compression_ratio = vae.config.spatial_compression_ratio
    
    # 处理 fix_sample_size
    if args.fix_sample_size is not None and args.enable_bucket:
        args.video_sample_size = max(max(args.fix_sample_size), args.video_sample_size)
        args.image_sample_size = max(max(args.fix_sample_size), args.image_sample_size)
        args.training_with_video_token_length = False
        args.random_hw_adapt = False
    
    # 创建数据集
    train_dataset = ImageVideoDataset(
        args.train_data_meta,
        args.train_data_dir,
        video_sample_size=args.video_sample_size,
        video_sample_stride=args.video_sample_stride,
        video_sample_n_frames=args.video_sample_n_frames,
        video_repeat=args.video_repeat,
        image_sample_size=args.image_sample_size,
        enable_bucket=args.enable_bucket,
        enable_inpaint=True,  # I2V 模式需要 inpaint
        height=args.height,
        width=args.width,
    )
    
    # 创建 DataLoader
    if args.enable_bucket:
        # 使用 bucket sampler
        aspect_ratio_sample_size = {
            key: [x / 512 * args.video_sample_size for x in ASPECT_RATIO_512[key]]
            for key in ASPECT_RATIO_512.keys()
        }
        
        batch_sampler_generator = torch.Generator().manual_seed(args.seed)
        batch_sampler = AspectRatioBatchImageVideoSampler(
            sampler=RandomSampler(train_dataset, generator=batch_sampler_generator),
            dataset=train_dataset.dataset,
            batch_size=args.train_batch_size,
            train_folder=args.train_data_dir,
            drop_last=True,
            aspect_ratios=aspect_ratio_sample_size,
        )
        
        # Collate function
        def collate_fn(examples):
            target_token_length = args.video_sample_n_frames * args.token_sample_size * args.token_sample_size
            
            new_examples = {}
            new_examples["target_token_length"] = target_token_length
            new_examples["pixel_values"] = []
            new_examples["text"] = []
            new_examples["mask_pixel_values"] = []
            new_examples["mask"] = []
            new_examples["clip_pixel_values"] = []
            
            # 获取下采样比例
            pixel_value = examples[0]["pixel_values"]
            data_type = examples[0]["data_type"]
            f, h, w, c = np.shape(pixel_value)
            
            if data_type == 'image':
                random_downsample_ratio = 1 if not args.random_hw_adapt else get_random_downsample_ratio(
                    args.image_sample_size,
                    image_ratio=[args.image_sample_size / args.video_sample_size],
                    rng=rng
                )
                aspect_ratio_sample_size_local = {
                    key: [x / 512 * args.image_sample_size / random_downsample_ratio for x in ASPECT_RATIO_512[key]]
                    for key in ASPECT_RATIO_512.keys()
                }
                batch_video_length = args.video_sample_n_frames + sample_n_frames_bucket_interval
            else:
                if args.random_hw_adapt:
                    random_downsample_ratio = get_random_downsample_ratio(args.video_sample_size, rng=rng)
                    batch_video_length = args.video_sample_n_frames + sample_n_frames_bucket_interval
                else:
                    random_downsample_ratio = 1
                    batch_video_length = args.video_sample_n_frames + sample_n_frames_bucket_interval
                
                aspect_ratio_sample_size_local = {
                    key: [x / 512 * args.video_sample_size / random_downsample_ratio for x in ASPECT_RATIO_512[key]]
                    for key in ASPECT_RATIO_512.keys()
                }
            
            # 确定采样尺寸
            if args.fix_sample_size is not None:
                fix_sample_size = [int(x / spatial_compression_ratio / 2) * spatial_compression_ratio * 2 
                                  for x in args.fix_sample_size]
            elif args.random_ratio_crop:
                if rng is None:
                    random_sample_size = {
                        key: [x / 512 * args.video_sample_size / random_downsample_ratio 
                             for x in ASPECT_RATIO_RANDOM_CROP_512[key]]
                        for key in ASPECT_RATIO_RANDOM_CROP_512.keys()
                    }
                    random_sample_size = random_sample_size[
                        np.random.choice(list(random_sample_size.keys()), p=ASPECT_RATIO_RANDOM_CROP_PROB)
                    ]
                else:
                    random_sample_size = {
                        key: [x / 512 * args.video_sample_size / random_downsample_ratio 
                             for x in ASPECT_RATIO_RANDOM_CROP_512[key]]
                        for key in ASPECT_RATIO_RANDOM_CROP_512.keys()
                    }
                    random_sample_size = random_sample_size[
                        rng.choice(list(random_sample_size.keys()), p=ASPECT_RATIO_RANDOM_CROP_PROB)
                    ]
                random_sample_size = [int(x / spatial_compression_ratio / 2) * spatial_compression_ratio * 2 
                                     for x in random_sample_size]
            else:
                closest_size, closest_ratio = get_closest_ratio(h, w, ratios=aspect_ratio_sample_size_local)
                closest_size = [int(x / spatial_compression_ratio / 2) * spatial_compression_ratio * 2 
                               for x in closest_size]
            
            # 处理每个样本
            for example in examples:
                if args.fix_sample_size is not None:
                    pixel_values = torch.from_numpy(example["pixel_values"]).permute(0, 3, 1, 2).contiguous()
                    pixel_values = pixel_values / 255.
                    transform = transforms.Compose([
                        transforms.Resize(fix_sample_size, interpolation=transforms.InterpolationMode.BILINEAR),
                        transforms.CenterCrop(fix_sample_size),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
                    ])
                elif args.random_ratio_crop:
                    pixel_values = torch.from_numpy(example["pixel_values"]).permute(0, 3, 1, 2).contiguous()
                    pixel_values = pixel_values / 255.
                    b, c, h, w = pixel_values.size()
                    th, tw = random_sample_size
                    if th / tw > h / w:
                        nh = int(th)
                        nw = int(w / h * nh)
                    else:
                        nw = int(tw)
                        nh = int(h / w * nw)
                    transform = transforms.Compose([
                        transforms.Resize([nh, nw]),
                        transforms.CenterCrop([int(x) for x in random_sample_size]),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
                    ])
                else:
                    pixel_values = torch.from_numpy(example["pixel_values"]).permute(0, 3, 1, 2).contiguous()
                    pixel_values = pixel_values / 255.
                    closest_size = list(map(lambda x: int(x), closest_size))
                    if closest_size[0] / h > closest_size[1] / w:
                        resize_size = closest_size[0], int(w * closest_size[0] / h)
                    else:
                        resize_size = int(h * closest_size[1] / w), closest_size[1]
                    transform = transforms.Compose([
                        transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR),
                        transforms.CenterCrop(closest_size),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
                    ])
                
                new_examples["pixel_values"].append(transform(pixel_values))
                new_examples["text"].append(example["text"])
                
                batch_video_length = int(min(batch_video_length, len(pixel_values)))
                batch_video_length = (batch_video_length - 1) // sample_n_frames_bucket_interval * sample_n_frames_bucket_interval + 1
                if batch_video_length <= 0:
                    batch_video_length = 1
                
                # 处理 mask
                mask = get_random_mask(new_examples["pixel_values"][-1].size(), image_start_only=True)
                mask_pixel_values = new_examples["pixel_values"][-1] * (1 - mask)
                new_examples["mask_pixel_values"].append(mask_pixel_values)
                new_examples["mask"].append(mask)
                
                clip_pixel_values = new_examples["pixel_values"][-1][0].permute(1, 2, 0).contiguous()
                clip_pixel_values = (clip_pixel_values * 0.5 + 0.5) * 255
                new_examples["clip_pixel_values"].append(clip_pixel_values)
            
            # 统一帧数
            new_examples["pixel_values"] = torch.stack([example[:batch_video_length] 
                                                        for example in new_examples["pixel_values"]])
            new_examples["mask_pixel_values"] = torch.stack([example[:batch_video_length] 
                                                             for example in new_examples["mask_pixel_values"]])
            new_examples["mask"] = torch.stack([example[:batch_video_length] 
                                                for example in new_examples["mask"]])
            new_examples["clip_pixel_values"] = torch.stack([example for example in new_examples["clip_pixel_values"]])
            
            # 文本编码
            if args.enable_text_encoder_in_dataloader:
                prompt_ids = tokenizer(
                    new_examples['text'],
                    max_length=args.tokenizer_max_length,
                    padding="max_length",
                    add_special_tokens=True,
                    truncation=True,
                    return_tensors="pt"
                )
                encoder_hidden_states = text_encoder(prompt_ids.input_ids)[0]
                new_examples['encoder_attention_mask'] = prompt_ids.attention_mask
                new_examples['encoder_hidden_states'] = encoder_hidden_states
            
            return new_examples
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            persistent_workers=True if args.dataloader_num_workers != 0 else False,
            num_workers=args.dataloader_num_workers,
        )
    else:
        # 使用简单的 sampler
        batch_sampler_generator = torch.Generator().manual_seed(args.seed)
        batch_sampler = ImageVideoSampler(
            RandomSampler(train_dataset, generator=batch_sampler_generator),
            train_dataset,
            args.train_batch_size
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            persistent_workers=True if args.dataloader_num_workers != 0 else False,
            num_workers=args.dataloader_num_workers,
        )
    
    # ====================================
    # 4. 优化器和学习率调度器
    # ====================================
    logger.info("创建优化器...")
    optimizer = torch.optim.AdamW(
        transformer.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
        eps=args.adam_epsilon,
    )
    
    # 计算训练步数
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    
    # ====================================
    # 5. Accelerate 准备
    # ====================================
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )
    
    # 将冻结的模型移到设备上
    vae.to(accelerator.device, dtype=weight_dtype)
    if not args.enable_text_encoder_in_dataloader:
        text_encoder.to(accelerator.device, dtype=weight_dtype)
    
    # 重新计算训练步数
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    
    # ====================================
    # 6. 训练循环
    # ====================================
    logger.info("***** 开始训练 *****")
    logger.info(f"  样本数量 = {len(train_dataset)}")
    logger.info(f"  训练轮数 = {args.num_train_epochs}")
    logger.info(f"  批次大小 = {args.train_batch_size}")
    logger.info(f"  梯度累积步数 = {args.gradient_accumulation_steps}")
    logger.info(f"  总优化步数 = {args.max_train_steps}")
    
    global_step = 0
    first_epoch = 0
    
    # 多 CUDA 流（可选）
    if args.multi_stream:
        vae_stream_1 = torch.cuda.Stream()
        vae_stream_2 = torch.cuda.Stream()
    else:
        vae_stream_1 = None
        vae_stream_2 = None
    
    # 计算 boundary 相关参数
    boundary = config['transformer_additional_kwargs'].get('boundary', 0.900)
    split_timesteps = args.train_sampling_steps * boundary
    differences = torch.abs(noise_scheduler.timesteps - split_timesteps)
    closest_index = torch.argmin(differences).item()
    
    if args.boundary_type == "high":
        start_num_idx = 0
        train_sampling_steps = closest_index
    elif args.boundary_type == "low":
        start_num_idx = closest_index
        train_sampling_steps = args.train_sampling_steps - closest_index
    else:
        start_num_idx = 0
        train_sampling_steps = args.train_sampling_steps
    
    idx_sampling = DiscreteSampling(train_sampling_steps, start_num_idx=start_num_idx, 
                                    uniform_sampling=args.uniform_sampling)
    
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        train_loss = 0.0
        
        if args.enable_bucket:
            batch_sampler.sampler.generator = torch.Generator().manual_seed(args.seed + epoch)
        
        for step, batch in enumerate(train_dataloader):
            # 第一个epoch第一个step保存样本检查
            if epoch == first_epoch and step == 0:
                pixel_values, texts = batch['pixel_values'].cpu(), batch['text']
                val_height, val_width = batch["pixel_values"].size()[-2], batch["pixel_values"].size()[-1]
                
                pixel_values_vis = rearrange(pixel_values, "b f c h w -> b c f h w")
                os.makedirs(os.path.join(args.output_dir, "sanity_check"), exist_ok=True)
                for idx, (pixel_value, text) in enumerate(zip(pixel_values_vis, texts)):
                    pixel_value = pixel_value[None, ...]
                    gif_name = '-'.join(text.replace('/', '').split()[:10]) if text else f'{global_step}-{idx}'
                    save_videos_grid(pixel_value, f"{args.output_dir}/sanity_check/{gif_name[:10]}.gif", rescale=True)
                
                # 保存 mask 相关可视化
                clip_pixel_values = batch['clip_pixel_values'].cpu()
                mask_pixel_values = batch['mask_pixel_values'].cpu()
                mask_pixel_values = rearrange(mask_pixel_values, "b f c h w -> b c f h w")
                for idx, (clip_pixel_value, pixel_value, text) in enumerate(zip(clip_pixel_values, mask_pixel_values, texts)):
                    pixel_value = pixel_value[None, ...]
                    Image.fromarray(np.uint8(clip_pixel_value)).save(
                        f"{args.output_dir}/sanity_check/clip_{gif_name[:10] if text else f'{global_step}-{idx}'}.png"
                    )
                    save_videos_grid(
                        pixel_value,
                        f"{args.output_dir}/sanity_check/mask_{gif_name[:10] if text else f'{global_step}-{idx}'}.gif",
                        rescale=True
                    )
            
            with accelerator.accumulate(transformer):
                # ---------------------------
                # 准备数据
                # ---------------------------
                pixel_values = batch["pixel_values"].to(weight_dtype)
                mask_pixel_values = batch["mask_pixel_values"].to(weight_dtype)
                mask = batch["mask"].to(weight_dtype)
                
                # 可选：自动调整 batch size
                if args.auto_tile_batch_size and args.training_with_video_token_length:
                    if args.video_sample_n_frames * args.token_sample_size * args.token_sample_size // 16 >= \
                       pixel_values.size()[1] * pixel_values.size()[3] * pixel_values.size()[4]:
                        pixel_values = torch.tile(pixel_values, (4, 1, 1, 1, 1))
                        mask_pixel_values = torch.tile(mask_pixel_values, (4, 1, 1, 1, 1))
                        mask = torch.tile(mask, (4, 1, 1, 1, 1))
                        if args.enable_text_encoder_in_dataloader:
                            batch['encoder_hidden_states'] = torch.tile(batch['encoder_hidden_states'], (4, 1, 1))
                            batch['encoder_attention_mask'] = torch.tile(batch['encoder_attention_mask'], (4, 1))
                        else:
                            batch['text'] = batch['text'] * 4
                    elif args.video_sample_n_frames * args.token_sample_size * args.token_sample_size // 4 >= \
                         pixel_values.size()[1] * pixel_values.size()[3] * pixel_values.size()[4]:
                        pixel_values = torch.tile(pixel_values, (2, 1, 1, 1, 1))
                        mask_pixel_values = torch.tile(mask_pixel_values, (2, 1, 1, 1, 1))
                        mask = torch.tile(mask, (2, 1, 1, 1, 1))
                        if args.enable_text_encoder_in_dataloader:
                            batch['encoder_hidden_states'] = torch.tile(batch['encoder_hidden_states'], (2, 1, 1))
                            batch['encoder_attention_mask'] = torch.tile(batch['encoder_attention_mask'], (2, 1))
                        else:
                            batch['text'] = batch['text'] * 2
                
                # 可选：随机帧裁剪
                if args.random_frame_crop:
                    def _create_special_list(length):
                        if length == 1:
                            return [1.0]
                        if length >= 2:
                            last_element = 0.90
                            remaining_sum = 1.0 - last_element
                            other_elements_value = remaining_sum / (length - 1)
                            special_list = [other_elements_value] * (length - 1) + [last_element]
                            return special_list
                    
                    select_frames = [_tmp for _tmp in list(range(
                        sample_n_frames_bucket_interval + 1,
                        args.video_sample_n_frames + sample_n_frames_bucket_interval,
                        sample_n_frames_bucket_interval
                    ))]
                    select_frames_prob = np.array(_create_special_list(len(select_frames)))
                    
                    if len(select_frames) != 0:
                        if rng is None:
                            temp_n_frames = np.random.choice(select_frames, p=select_frames_prob)
                        else:
                            temp_n_frames = rng.choice(select_frames, p=select_frames_prob)
                    else:
                        temp_n_frames = 1
                    
                    temp_n_frames = (temp_n_frames - 1) // sample_n_frames_bucket_interval + 1
                    pixel_values = pixel_values[:, :temp_n_frames, :, :]
                    mask_pixel_values = mask_pixel_values[:, :temp_n_frames, :, :]
                    mask = mask[:, :temp_n_frames, :, :]
                
                # ---------------------------
                # VAE 编码
                # ---------------------------
                with torch.no_grad():
                    def _batch_encode_vae(pixel_values):
                        pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                        bs = args.vae_mini_batch
                        new_pixel_values = []
                        for i in range(0, pixel_values.shape[0], bs):
                            pixel_values_bs = pixel_values[i:i + bs]
                            pixel_values_bs = vae.encode(pixel_values_bs)[0]
                            pixel_values_bs = pixel_values_bs.sample()
                            new_pixel_values.append(pixel_values_bs)
                        return torch.cat(new_pixel_values, dim=0)
                    
                    if vae_stream_1 is not None:
                        vae_stream_1.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(vae_stream_1):
                            latents = _batch_encode_vae(pixel_values)
                    else:
                        latents = _batch_encode_vae(pixel_values)
                    
                    # 编码 mask
                    mask = rearrange(mask, "b f c h w -> b c f h w")
                    mask = torch.concat([
                        torch.repeat_interleave(mask[:, :, 0:1], repeats=4, dim=2),
                        mask[:, :, 1:]
                    ], dim=2)
                    mask = mask.view(mask.shape[0], mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4])
                    mask = mask.transpose(1, 2)
                    mask = resize_mask(1 - mask, latents)
                    
                    # 编码 mask latents
                    mask_latents = _batch_encode_vae(mask_pixel_values)
                    if vae_stream_2 is not None:
                        torch.cuda.current_stream().wait_stream(vae_stream_2)
                    
                    inpaint_latents = torch.concat([mask, mask_latents], dim=1)
                    
                    # **I2V 核心逻辑：处理 t2v flag**
                    # 当 mask 全为 1 时，有 90% 概率转为 t2v 任务
                    t2v_flag = [(_mask == 1).all() for _mask in mask]
                    new_t2v_flag = []
                    for _mask in t2v_flag:
                        if _mask and np.random.rand() < 0.90:
                            new_t2v_flag.append(0)
                        else:
                            new_t2v_flag.append(1)
                    t2v_flag = torch.from_numpy(np.array(new_t2v_flag)).to(accelerator.device, dtype=weight_dtype)
                    inpaint_latents = t2v_flag[:, None, None, None, None] * inpaint_latents
                
                # 等待 VAE 编码完成
                if vae_stream_1 is not None:
                    torch.cuda.current_stream().wait_stream(vae_stream_1)
                
                # 文本编码
                if args.enable_text_encoder_in_dataloader:
                    prompt_embeds = batch['encoder_hidden_states'].to(device=latents.device)
                else:
                    with torch.no_grad():
                        prompt_ids = tokenizer(
                            batch['text'],
                            padding="max_length",
                            max_length=args.tokenizer_max_length,
                            truncation=True,
                            add_special_tokens=True,
                            return_tensors="pt"
                        )
                        text_input_ids = prompt_ids.input_ids
                        prompt_attention_mask = prompt_ids.attention_mask
                        
                        seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long()
                        prompt_embeds = text_encoder(
                            text_input_ids.to(latents.device),
                            attention_mask=prompt_attention_mask.to(latents.device)
                        )[0]
                        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
                
                # ---------------------------
                # Flow Matching 训练
                # ---------------------------
                bsz, channel, num_frames, height, width = latents.size()
                noise = torch.randn(latents.size(), device=latents.device, generator=torch_rng, dtype=weight_dtype)
                
                # 采样时间步
                indices = idx_sampling(bsz, generator=torch_rng, device=latents.device)
                indices = indices.long().cpu()
                timesteps = noise_scheduler.timesteps[indices].to(device=latents.device)
                
                # 计算 sigma
                def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
                    sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
                    schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
                    timesteps = timesteps.to(accelerator.device)
                    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
                    
                    sigma = sigmas[step_indices].flatten()
                    while len(sigma.shape) < n_dim:
                        sigma = sigma.unsqueeze(-1)
                    return sigma
                
                sigmas = get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
                noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
                target = noise - latents
                
                # 计算 seq_len
                target_shape = (vae.latent_channels, num_frames, width, height)
                seq_len = math.ceil(
                    (target_shape[2] * target_shape[3]) /
                    (accelerator.unwrap_model(transformer).config.patch_size[1] * 
                     accelerator.unwrap_model(transformer).config.patch_size[2]) *
                    target_shape[1]
                )
                
                # ---------------------------
                # 前向传播
                # ---------------------------
                with torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=accelerator.device):
                    noise_pred = transformer(
                        x=noisy_latents,
                        context=prompt_embeds,
                        t=timesteps,
                        seq_len=seq_len,
                        y=inpaint_latents,
                    )
                
                # ---------------------------
                # 计算损失
                # ---------------------------
                def custom_mse_loss(noise_pred, target, weighting=None, threshold=50):
                    noise_pred = noise_pred.float()
                    target = target.float()
                    diff = noise_pred - target
                    mse_loss = F.mse_loss(noise_pred, target, reduction='none')
                    mask = (diff.abs() <= threshold).float()
                    masked_loss = mse_loss * mask
                    if weighting is not None:
                        masked_loss = masked_loss * weighting
                    final_loss = masked_loss.mean()
                    return final_loss
                
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                loss = custom_mse_loss(noise_pred.float(), target.float(), weighting.float())
                loss = loss.mean()
                
                # 收集损失
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                
                # 反向传播
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # ---------------------------
            # 日志和保存
            # ---------------------------
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                if global_step % args.logging_steps == 0:
                    logs = {"train_loss": train_loss, "lr": lr_scheduler.get_last_lr()[0]}
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)
                    train_loss = 0.0
                
                # 保存检查点
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # 限制检查点数量
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]
                                
                                logger.info(f"移除 {len(removing_checkpoints)} 个旧检查点")
                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)
                        
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(save_path, exist_ok=True)
                        
                        # 保存 transformer
                        unwrapped_transformer = accelerator.unwrap_model(transformer)
                        unwrapped_transformer.save_pretrained(os.path.join(save_path, "transformer"))
                        
                        # 保存 optimizer 和 scheduler
                        torch.save(optimizer.state_dict(), os.path.join(save_path, "optimizer.pt"))
                        torch.save(lr_scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))
                        
                        logger.info(f"保存检查点到 {save_path}")
                
                # 验证
                if accelerator.is_main_process:
                    if args.validation_prompts is not None and global_step % args.validation_steps == 0:
                        log_validation(
                            vae, text_encoder, tokenizer, transformer,
                            config, args, accelerator, weight_dtype,
                            global_step, val_height, val_width
                        )
            
            if global_step >= args.max_train_steps:
                break
    
    # 保存最终模型
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        os.makedirs(final_save_path, exist_ok=True)
        
        unwrapped_transformer = accelerator.unwrap_model(transformer)
        unwrapped_transformer.save_pretrained(os.path.join(final_save_path, "transformer"))
        
        logger.info(f"训练完成！最终模型保存到 {final_save_path}")
    
    accelerator.end_training()


if __name__ == "__main__":
    main()