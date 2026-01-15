import os
import sys
import numpy as np
import torch
import torch.multiprocessing as mp
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image
import pandas as pd
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.dist import set_multi_gpus_devices, shard_model
from videox_fun.models import (AutoencoderKLWan, AutoencoderKLWan3_8, AutoTokenizer, CLIPModel,
                              WanT5EncoderModel, Wan2_2Transformer3DModel)
from videox_fun.models.cache_utils import get_teacache_coefficients
from videox_fun.pipeline import Wan2_2I2VPipeline
from videox_fun.utils.fp8_optimization import (convert_model_weight_to_float8, replace_parameters_by_name,
                                              convert_weight_dtype_wrapper)
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import (filter_kwargs, get_image_to_video_latent, save_videos_grid)
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


def build_wan22_pipeline(
    config_path: str,
    model_name: str,
    vae_path: str | None = None,
    transformer_path: str | None = None,
    transformer_high_path: str | None = None,
    weight_dtype: torch.dtype = torch.bfloat16,
    device: str | None = None,
    ulysses_degree: int = 1,
    ring_degree: int = 1,
    fsdp_dit: bool = False,
    fsdp_text_encoder: bool = True,
    compile_dit: bool = False,
    GPU_memory_mode: str | None = None,
    sampler_name: str = "Flow_Unipc",
    lora_weight=1.0, lora_high_weight=1.0,
    lora_low=None,
    lora_high=None,
    **scheduler_extra,
):
    if device is None:
        device = set_multi_gpus_devices(ulysses_degree, ring_degree)

    cfg = OmegaConf.load(config_path)
    boundary = cfg['transformer_additional_kwargs'].get('boundary', 0.900)

    low_path  = os.path.join(model_name, cfg['transformer_additional_kwargs'].get(
                              'transformer_low_noise_model_subpath',  'transformer'))
    high_path = os.path.join(model_name, cfg['transformer_additional_kwargs'].get(
                              'transformer_high_noise_model_subpath', 'transformer'))

    transformer   = Wan2_2Transformer3DModel.from_pretrained(
        low_path,
        transformer_additional_kwargs=OmegaConf.to_container(cfg['transformer_additional_kwargs']),
        low_cpu_mem_usage=True, torch_dtype=weight_dtype,
    )
    transformer_2 = Wan2_2Transformer3DModel.from_pretrained(
        high_path,
        transformer_additional_kwargs=OmegaConf.to_container(cfg['transformer_additional_kwargs']),
        low_cpu_mem_usage=True, torch_dtype=weight_dtype,
    )

    def maybe_load(sd_path, target):
        if sd_path is None:  return
        print(f"Loading custom checkpoint: {sd_path}")
        load_fn = torch.load if not sd_path.endswith("safetensors") else \
                  (lambda p: __import__("safetensors.torch").torch.load_file(p))
        sd = load_fn(sd_path)
        sd = sd["state_dict"] if "state_dict" in sd else sd
        missing, unexpected = target.load_state_dict(sd, strict=False)
        print(f"  ➜ missing {len(missing)} · unexpected {len(unexpected)}")

    maybe_load(transformer_path, transformer)
    maybe_load(transformer_high_path, transformer_2)

    AEClass = AutoencoderKLWan3_8 if cfg['vae_kwargs']['vae_type'] == 'Wan3_8' else AutoencoderKLWan
    vae = AEClass.from_pretrained(
        os.path.join(model_name, cfg['vae_kwargs'].get('vae_subpath', 'vae')),
        additional_kwargs=OmegaConf.to_container(cfg['vae_kwargs'])
    ).to(weight_dtype)
    maybe_load(vae_path, vae)

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(model_name, cfg['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer'))
    )
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(model_name, cfg['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
        additional_kwargs=OmegaConf.to_container(cfg['text_encoder_kwargs']),
        low_cpu_mem_usage=True, torch_dtype=weight_dtype,
    ).eval()

    sched_map = {
        "Flow":       FlowMatchEulerDiscreteScheduler,
        "Flow_Unipc": FlowUniPCMultistepScheduler,
        "Flow_DPM++": FlowDPMSolverMultistepScheduler,
    }
    Schd = sched_map[sampler_name]
    if sampler_name in ["Flow_Unipc", "Flow_DPM++"]:
        cfg['scheduler_kwargs']['shift'] = 1
    scheduler_cfg = dict(OmegaConf.to_container(cfg['scheduler_kwargs']))
    scheduler_cfg.update(scheduler_extra)
    scheduler = Schd(**filter_kwargs(Schd, scheduler_cfg))

    pipe = Wan2_2I2VPipeline(
        transformer=transformer, transformer_2=transformer_2,
        vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=scheduler,
    )
    pipe = pipe.to(device)
    
    if lora_low:
        pipe = merge_lora(pipe, lora_low, lora_weight, device=device)
        pipe = merge_lora(pipe, lora_high, lora_high_weight,
                          device=device, sub_transformer_name="transformer_2")
    return pipe, vae, boundary, device


def infer_video(
    pipeline, vae, boundary, device, *,
    sample_size: list[int],
    video_length: int,
    validation_image_start: str,
    prompt: str,
    save_path: str,
    negative_prompt: str = "色调艳丽，过曝，静态…",
    fps: int = 16,
    seed: int = 42,
    guidance_scale: float = 6.0,
    num_inference_steps: int = 50,
    shift: int = 5,
    lora_low=None, lora_high=None,
    lora_weight=1.0, lora_high_weight=1.0,
    enable_riflex=False, riflex_k=6,
):
    generator = torch.Generator(device=device).manual_seed(seed)

    with torch.no_grad():
        if video_length != 1:
            video_length = ((video_length - 1) //
                            vae.config.temporal_compression_ratio *
                            vae.config.temporal_compression_ratio) + 1
        latent_frames = (video_length - 1) // vae.config.temporal_compression_ratio + 1

        if enable_riflex:
            pipeline.transformer.enable_riflex(k=riflex_k, L_test=latent_frames)
            pipeline.transformer_2.enable_riflex(k=riflex_k, L_test=latent_frames)

        video_latent, mask_latent, _ = get_image_to_video_latent(
            validation_image_start, None,
            video_length=video_length, sample_size=sample_size
        )

        sample = pipeline(
            prompt,
            num_frames=video_length,
            negative_prompt=negative_prompt,
            height=sample_size[0], width=sample_size[1],
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            boundary=boundary,
            video=video_latent, mask_video=mask_latent,
            shift=shift,
        ).videos

    os.makedirs(save_path, exist_ok=True)
    image_basename = os.path.splitext(os.path.basename(validation_image_start))[0]

    if video_length == 1:
        out_png = os.path.join(save_path, image_basename + ".png")
        img = sample[0, :, 0].permute(1, 2, 0).cpu().numpy() * 255
        Image.fromarray(img.astype(np.uint8)).save(out_png)
        return out_png
    else:
        out_mp4 = os.path.join(save_path, image_basename + ".mp4")
        save_videos_grid(sample, out_mp4, fps=fps)
        return out_mp4


def worker_fn(rank: int, world_size: int, args, paths: list, prompts: list):
    """Worker process: loads model on assigned GPU and processes its data shard."""
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)
    
    # Shard data: each GPU gets indices [rank, rank+world_size, rank+2*world_size, ...]
    indices = list(range(rank, len(paths), world_size))
    local_paths = [paths[i] for i in indices]
    local_prompts = [prompts[i] for i in indices]
    
    print(f"[GPU {rank}] Processing {len(local_paths)} samples")
    
    # Build pipeline on this GPU
    pipe, vae, boundary, _ = build_wan22_pipeline(
        config_path=args.config_path,
        model_name=args.model_name,
        lora_low=args.lora_low,
        lora_high=args.lora_high,
        device=device,
    )
    
    # Parse resolution
    if args.resolution:
        height, width = map(int, args.resolution.split('x'))
    else:
        height, width = args.height, args.width
    sample_size = [height, width]
    
    negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    
    for i, (path, prompt) in enumerate(zip(local_paths, local_prompts)):
        if not os.path.isfile(path):
            print(f"[GPU {rank}] Warning: Image not found: {path}")
            continue
        
        video_path = os.path.join(
            args.output_path,
            os.path.splitext(os.path.basename(path))[0] + ".mp4"
        )
        if os.path.exists(video_path):
            print(f"[GPU {rank}] Skip existing: {video_path}")
            continue
        
        print(f"[GPU {rank}] {i+1}/{len(local_paths)}: {os.path.basename(path)}")
        infer_video(
            pipe, vae, boundary, device,
            sample_size=sample_size,
            video_length=81,
            validation_image_start=path,
            prompt=prompt,
            save_path=args.output_path,
            negative_prompt=negative_prompt,
        )


def parse_args():
    parser = argparse.ArgumentParser(description='Multi-GPU video generation')
    parser.add_argument('--model_name', type=str, default="./Models/Wan2.2-I2V-A14B")
    parser.add_argument('--lora_low', type=str, 
        default="./Models/Lora/10_LargeMixedDatset_wan_14bLow_f81_LongCaption_StartMatch_run_r128_a128_3obj_Longrun_B4/checkpoint-600.safetensors")
    parser.add_argument('--lora_high', type=str,
        default="./Models/Lora/10_LargeMixedDatset_wan_14bHigh_f81_LongCaption_StartMatch_run_r128_a128_3obj_Longrun_B4/checkpoint-600.safetensors")
    parser.add_argument('--config_path', type=str, default="./VideoX-Fun/config/wan2.2/wan_civitai_i2v.yaml")
    parser.add_argument('--data_csv', type=str, default='./Data/combined_first_frames/0-data.csv')
    parser.add_argument('--dataset_dir', type=str, default='./Data/combined_first_frames')
    parser.add_argument('--output_path', type=str, default="./output/ffgo_eval")
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--resolution', type=str, default=None)
    parser.add_argument('--num_gpus', type=int, default=None, 
        help='Number of GPUs to use (default: all available)')
    return parser.parse_args()


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    args = parse_args()
    
    # Determine number of GPUs
    num_gpus = args.num_gpus or torch.cuda.device_count()
    num_gpus = min(num_gpus, 8, torch.cuda.device_count())
    print(f"Using {num_gpus} GPUs")
    
    # Load data
    df = pd.read_csv(args.data_csv)
    prompts = list(df['prompt'])
    paths = [os.path.join(args.dataset_dir, p) for p in df['image_path']]
    
    os.makedirs(args.output_path, exist_ok=True)
    
    # Spawn workers
    mp.spawn(worker_fn, args=(num_gpus, args, paths, prompts), nprocs=num_gpus, join=True)
    
    print("All done!")