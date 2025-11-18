import os
import sys

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image

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
from videox_fun.utils.utils import (filter_kwargs, get_image_to_video_latent,
                                   save_videos_grid)
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

# GPU memory mode, which can be chosen in [model_full_load, model_full_load_and_qfloat8, model_cpu_offload, model_cpu_offload_and_qfloat8, sequential_cpu_offload].
# model_full_load means that the entire model will be moved to the GPU.
# 
# model_full_load_and_qfloat8 means that the entire model will be moved to the GPU,
# and the transformer model has been quantized to float8, which can save more GPU memory. 
# 
# model_cpu_offload means that the entire model will be moved to the CPU after use, which can save some GPU memory.
# 
# model_cpu_offload_and_qfloat8 indicates that the entire model will be moved to the CPU after use, 
# and the transformer model has been quantized to float8, which can save more GPU memory. 
# 
# sequential_cpu_offload means that each layer of the model will be moved to the CPU after use, 
# resulting in slower speeds but saving a large amount of GPU memory.
# GPU_memory_mode     = "sequential_cpu_offload"
GPU_memory_mode     = None
# Multi GPUs config
# Please ensure that the product of ulysses_degree and ring_degree equals the number of GPUs used. 
# For example, if you are using 8 GPUs, you can set ulysses_degree = 2 and ring_degree = 4.
# If you are using 1 GPU, you can set ulysses_degree = 1 and ring_degree = 1.
ulysses_degree      = 1
ring_degree         = 1
# Use FSDP to save more GPU memory in multi gpus.
fsdp_dit            = False
fsdp_text_encoder   = True
# Compile will give a speedup in fixed resolution and need a little GPU memory. 
# The compile_dit is not compatible with the fsdp_dit and sequential_cpu_offload.
compile_dit         = False

# TeaCache config
enable_teacache     = False
# Recommended to be set between 0.05 and 0.30. A larger threshold can cache more steps, speeding up the inference process, 
# but it may cause slight differences between the generated content and the original content.
# # --------------------------------------------------------------------------------------------------- #
# | Model Name          | threshold | Model Name          | threshold |
# | Wan2.2-T2V-A14B     | 0.10~0.15 | Wan2.2-I2V-A14B     | 0.15~0.20 |
# # --------------------------------------------------------------------------------------------------- #
teacache_threshold  = 0.10
# The number of steps to skip TeaCache at the beginning of the inference process, which can
# reduce the impact of TeaCache on generated video quality.
num_skip_start_steps = 5
# Whether to offload TeaCache tensors to cpu to save a little bit of GPU memory.
teacache_offload    = False

# Skip some cfg steps in inference
# Recommended to be set between 0.00 and 0.25
cfg_skip_ratio      = 0

# Riflex config
enable_riflex       = False
# Index of intrinsic frequency
riflex_k            = 6

# Config and model path
config_path         = "/workspace/Project/VideoRAG/VideoX-Fun/config/wan2.2/wan_civitai_i2v.yaml"
# model path
model_name          = "/workspace/hfhome/hub/models--Wan-AI--Wan2.2-I2V-A14B/snapshots/206a9ee1b7bfaaf8f7e4d81335650533490646a3"

# Choose the sampler in "Flow", "Flow_Unipc", "Flow_DPM++"
sampler_name        = "Flow_Unipc"
# [NOTE]: Noise schedule shift parameter. Affects temporal dynamics. 
# Used when the sampler is in "Flow_Unipc", "Flow_DPM++".
shift               = 5

# Load pretrained model if need
# The transformer_path is used for low noise model, the transformer_high_path is used for high noise model.
transformer_path        = None
transformer_high_path   = None
vae_path                = None
# Load lora model if need
# The lora_path is used for low noise model, the lora_high_path is used for high noise model.
# lora_path               = "/workspace/Models/07_smallMixedDatset_wan_14bLow_f81_LongCaption_StartMatch_run_r128_a128_3obj_Longrun_B4/checkpoint-500.safetensors"
# lora_high_path          = "/workspace/Models/07_smallMixedDatset_wan_14bHigh_f81_LongCaption_StartMatch_run_r128_a128_3obj_Longrun_B4/checkpoint-500.safetensors"

# lora_path = "/workspace/Models/08_smallMixedDatset_wan_14bLow_f81_LongCaption_StartMatch_run_r128_a128_3obj_Longrun_B4/checkpoint-600.safetensors"
# lora_high_path = "/workspace/Models/08_smallMixedDatset_wan_14bHigh_f81_LongCaption_StartMatch_run_r128_a128_3obj_Longrun_B4/checkpoint-600.safetensors"

lora_path               = "/workspace/Models/10_LargeMixedDatset_wan_14bLow_f81_LongCaption_StartMatch_run_r128_a128_3obj_Longrun_B4/checkpoint-600.safetensors"
lora_high_path          = "/workspace/Models/10_LargeMixedDatset_wan_14bHigh_f81_LongCaption_StartMatch_run_r128_a128_3obj_Longrun_B4/checkpoint-600.safetensors"

# Other params
# sample_size         = [768, 1344]
sample_size         = [480, 640]
# sample_size = [720, 1280]
video_length        = 81
fps                 = 16

# Use torch.float16 if GPU does not support torch.bfloat16
# ome graphics cards, such as v100, 2080ti, do not support torch.bfloat16
weight_dtype            = torch.bfloat16
# If you want to generate from text, please set the validation_image_start = None and validation_image_end = None
# validation_image_start  =  "/workspace/Data/01_test_combined/8_combined_lzx_instruct1.png"
# validation_image_start  =  "/workspace/Data/01_zongxia_test/combined_4_2.png"
validation_image_start  =  "/workspace/Data/01_zongxia_test/combined_12.png"


# validation_image_start  =  "/workspace/Data/01_test_combined/9_combined.png"



# prompts
### 20
# prompt = "ad23r2 the camera view suddenly changes. The video opens from a first-person driver's perspective, navigating a multi-lane highway during a heavy downpour. The road is slick with rain, and visibility is poor due to fog and the spray from surrounding cars. Suddenly, a bright red Ferrari F8 Tributo aggressively accelerates from an adjacent lane. The sports car swerves sharply, cutting directly in front of the camera's vehicle with very little space. The camera focuses on the rear of the red Ferrari, its distinctive twin taillights glowing, as it kicks up a massive plume of water and road spray, momentarily blinding the driver before it speeds off into the misty conditions."

### 19
# prompt = "ad23r2 the camera view suddenly changes. The video begins with an establishing shot of a bright, modern conference room with a long white table. Seated across from each other are Albert Einstein, with his iconic wild white hair and tweed jacket, and Mahatma Gandhi, clad in his simple white robes and round glasses. The camera alternates between medium shots of their interaction and close-ups focusing on their expressions. Einstein appears animated, gesturing with his hands as he speaks passionately about science, his face growing somber as the topic turns to war. Gandhi listens intently, his demeanor calm and thoughtful, before leaning forward to offer a quiet, considered response."

### 18
# prompt = "ad23r2 the camera view suddenly changes. The video opens on a dramatic, wide-angle shot of a desolate badlands landscape, dominated by eroded mesas and grey, pointed hills under a vibrant sunset sky filled with purple and orange clouds. The camera then focuses on the upper bodies of two massive figures: the blonde, heavily-muscled Armored Titan on the left and the dark-haired, leaner Attack Titan on the right. Both Titans, shown from the chest up, advance side-by-side, moving steadily and menacingly towards the camera. Close-up shots capture their intense, grimacing expressions as they press forward against the dramatic backdrop."

### 17
# prompt = "ad23r2 the camera view suddenly changes. A cinematic wide shot establishes the iconic red sandstone buttes and arid desert floor of Monument Valley, Utah. In the foreground, a realistically rendered Super Saiyan warrior, featuring spiky blond hair, a halo, and a Metamoran vest, crouches in a dynamic pose, charging a glowing blue energy blast in his outstretched hand. Facing him, a realistic depiction of Saitama from One-Punch Man stands impassively in his yellow suit, red gloves, and white cape, arms firmly crossed. The camera focuses on the intense standoff between the energy-wielding attacker and the stoic, unmoving hero as dust swirls around them, signaling the start of an epic battle."

### 16
# prompt = "ad23r2 the camera view suddenly changes. Cinematic medium shot tracking through a dense, lush jungle environment filled with large trees and green foliage. A large, powerfully built tiger with orange and black stripes walks steadily through the undergrowth. Riding bareback on the tiger is the muscular, shirtless man with long hair, wearing simple brown pants. He sits upright with a commanding and confident posture, moving effortlessly with the animal. The camera follows them, capturing the man's focused expression and his clear dominion over the powerful predator, portraying him as the undisputed king of the forest."

### 15
# prompt = "ad23r2 the camera view suddenly changes. A wide shot captures a presentation stage. On the left, Sam Altman stands wearing a grey sweatshirt and dark jeans. On the right, Microsoft CEO Satya Nadella stands wearing a blue t-shirt and dark jeans. Positioned in the center of the stage, between the two men, is a black and white humanoid robot. The robot begins to move, turning and walking smoothly towards Sam Altman on the left. The robot stops in front of him and extends its right hand. Sam Altman reaches out with his right hand, and they engage in a handshake, while Satya Nadella looks on from the right side of the stage."

### 14
# prompt = "ad23r2 the camera view suddenly changes. High-quality, realistic footage. A wide shot captures a white autonomous vehicle, a Lexus sedan with prominent sensors on its roof, driving along a curving paved test track. The car smoothly decelerates and comes to a complete stop. A small baby elephant enters the frame, walking slowly and deliberately across the paved road, passing directly in front of the stationary white car. The car remains motionless as the elephant completes its crossing and steps off the track onto the grass."


### 13
# prompt = "ad23r2 the camera view suddenly changes. A medium shot shows Akira Toriyama, smiling playfully while sitting. He holds the Super Saiyan Blue Goku action figure in his right hand and the Super Saiyan Blue Vegeta action figure in his left, both are around 20 centermeters tall. He brings the two figures close together, facing each other as if preparing for a showdown. The camera follows as Toriyama moves the Goku figure in a dynamic punching motion. He then maneuvers the Vegeta figure (in its arms-crossed pose) to \"block\" or \"parry\" the attack. He continues to move both figures in quick, energetic motions, simulating a high-speed battle in the air, looking down at them with an engaged and imaginative expression."

### 12
prompt = "ad23r2 the camera view suddenly changes. A medium shot features Akira Toriyama, smiling and wearing glasses, holding the red-haired Super Saiyan God Goku figure. The figure, depicted in its orange gi, stands just a bit larger than Akira's hands, appearing slightly larger than his hand as he holds it. Toriyama gestures towards the figure, looking directly at the camera with an enthusiastic expression, praising it by saying, \"it is very good,\" clearly promoting it for sale to the audience."

# prompt = "ad23r2 the camera view suddenly changes. Film quality, professional quality, rich details. Donald Trump and Kim Jong-un are positioned side-by-side in front of the White House. They exchange glances, and Kim Jong-un delivers a forceful face slap to Donald Trump, who then immediately kneels in apology."

# prompt = "ad23r2 the camera view suddenly changes. A cinematic, high-definition shot of the man in the light purple hoodie and the woman in the red coat standing together on a skyscraper rooftop. The sprawling city skyline is visible in the background, softly blurred by a shallow depth of field. A gentle breeze blows through the woman's long dark hair. They gaze into each other's eyes with deep affection, then lean in for a tender and emotional kiss. The camera slowly pushes in for a close-up on their faces, capturing the intimate and heartfelt moment as they fall in love."

# prompt = "ad23r2 the camera view suddenly changes. Film quality, professional quality, rich details. The man from Image 2 and the woman from Image 1 stand side-by-side. Both are looking forward, not at each other, maintaining the same face angles as in their respective images. Their faces hold soft, loving expressions, conveying a deep sense of shared affection in a tender moment."

# prompt = "ad23r2 the camera view suddenly changes. A high-quality, cinematic scene. A handsome young man in a pilot's uniform and a beautiful woman with long dark hair, wearing a vibrant red coat, stand intimately on a rooftop overlooking a modern city skyline during the golden hour. They gaze lovingly into each other's eyes. The camera slowly zooms in for a close-up as they lean in and share a tender, passionate kiss, capturing the romantic moment they fall in love."

# prompt = "ad23r2 the camera view suddenly changes. Film quality, professional quality, rich details. The video starts with the man and woman, both dressed in their respective attires, standing side-by-side on a stage with a red curtain as the backdrop. The camera captures them holding hands, highlighting their connection as a cute couple. The camera slowly zooms in on their faces."

# prompt = "ad23r2 the camera view suddenly changes. A medium-full shot captures a young Asian man and a beautiful Asian woman standing side-by-side on a beach, holding hands. The man is on the left, wearing a light teal, long-sleeved button-down shirt, black shorts, and has sunglasses resting on his head. The woman, on the right, is elegantly dressed in a formal strapless pink gown, complemented by a sparkling diamond necklace, earrings, and a bracelet. Behind them, the calm ocean with gentle waves meets a vast, pale blue sky. They both look towards the camera with soft smiles, portraying a romantic scene."

# prompt              = "ad23r2 the camera view suddenly changes.  A full shot of Donald Trump and Kim Jong-un standing side-by-side on the lawn in front of the White House. They slowly turn to face each other, making eye contact. Their expressions soften as they lean in and share a passionate, romantic kiss."

# prompt              = "ad23r2 the camera view suddenly changes.  Professional-quality video with rich details. The video features a charming Teddy Bear sitting in the pointed area in the background, sipping apple juice from a bottle using the hand, while delicately holds a vibrant red rose using the hand, admiring its beauty, perhaps as an offering or a gesture of affection."


# prompt= "ad23r2 the camera view suddenly changes.  A blue industrial robotic arm with multiple articulated joints and a gripper end-effector reaches down and picks up a turquoise blue electric SUV from a forest road. The robot arm, mounted on a fixed base, extends and rotates to grasp the vehicle carefully. The car, a modern EV with distinctive front styling, is lifted smoothly off the tree-lined asphalt road as the robotic arm's joints move in coordinated motion."


negative_prompt     = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

# Prompt with out "静止, 静止不动的画面"
# negative_prompt     = "色调艳丽，过曝，细节模糊不清，字幕，风格，作品，画作，画面, 整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，杂乱的背景，三条腿，背景人很多，倒着走"


guidance_scale      = 6.0
seed                = 42

# seed                = 62



num_inference_steps = 50
# The lora_weight is used for low noise model, the lora_high_weight is used for high noise model.
lora_weight         = 1
lora_high_weight    = 1
save_path           = "/workspace/zongxia_result/real_tests"

device = set_multi_gpus_devices(ulysses_degree, ring_degree)
config = OmegaConf.load(config_path)
boundary = config['transformer_additional_kwargs'].get('boundary', 0.900)

transformer = Wan2_2Transformer3DModel.from_pretrained(
    os.path.join(model_name, config['transformer_additional_kwargs'].get('transformer_low_noise_model_subpath', 'transformer')),
    transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)

transformer_2 = Wan2_2Transformer3DModel.from_pretrained(
    os.path.join(model_name, config['transformer_additional_kwargs'].get('transformer_high_noise_model_subpath', 'transformer')),
    transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)

if transformer_path is not None:
    print(f"From checkpoint: {transformer_path}")
    if transformer_path.endswith("safetensors"):
        from safetensors.torch import load_file, safe_open
        state_dict = load_file(transformer_path)
    else:
        state_dict = torch.load(transformer_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

    m, u = transformer.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

if transformer_high_path is not None:
    print(f"From checkpoint: {transformer_high_path}")
    if transformer_high_path.endswith("safetensors"):
        from safetensors.torch import load_file, safe_open
        state_dict = load_file(transformer_high_path)
    else:
        state_dict = torch.load(transformer_high_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

    m, u = transformer_2.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

# Get Vae
Chosen_AutoencoderKL = {
    "AutoencoderKLWan": AutoencoderKLWan,
    "AutoencoderKLWan3_8": AutoencoderKLWan3_8
}[config['vae_kwargs'].get('vae_type', 'AutoencoderKLWan')]
vae = Chosen_AutoencoderKL.from_pretrained(
    os.path.join(model_name, config['vae_kwargs'].get('vae_subpath', 'vae')),
    additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
).to(weight_dtype)

if vae_path is not None:
    print(f"From checkpoint: {vae_path}")
    if vae_path.endswith("safetensors"):
        from safetensors.torch import load_file, safe_open
        state_dict = load_file(vae_path)
    else:
        state_dict = torch.load(vae_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

    m, u = vae.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

# Get Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    os.path.join(model_name, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
)

# Get Text encoder
text_encoder = WanT5EncoderModel.from_pretrained(
    os.path.join(model_name, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
    additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)
text_encoder = text_encoder.eval()

# Get Scheduler
Chosen_Scheduler = scheduler_dict = {
    "Flow": FlowMatchEulerDiscreteScheduler,
    "Flow_Unipc": FlowUniPCMultistepScheduler,
    "Flow_DPM++": FlowDPMSolverMultistepScheduler,
}[sampler_name]
if sampler_name == "Flow_Unipc" or sampler_name == "Flow_DPM++":
    config['scheduler_kwargs']['shift'] = 1
scheduler = Chosen_Scheduler(
    **filter_kwargs(Chosen_Scheduler, OmegaConf.to_container(config['scheduler_kwargs']))
)

# Get Pipeline
pipeline = Wan2_2I2VPipeline(
    transformer=transformer,
    transformer_2=transformer_2,
    vae=vae,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    scheduler=scheduler,
)
if ulysses_degree > 1 or ring_degree > 1:
    from functools import partial
    transformer.enable_multi_gpus_inference()
    transformer_2.enable_multi_gpus_inference()
    if fsdp_dit:
        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
        pipeline.transformer = shard_fn(pipeline.transformer)
        pipeline.transformer_2 = shard_fn(pipeline.transformer_2)
        print("Add FSDP DIT")
    if fsdp_text_encoder:
        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
        pipeline.text_encoder = shard_fn(pipeline.text_encoder)
        print("Add FSDP TEXT ENCODER")

if compile_dit:
    for i in range(len(pipeline.transformer.blocks)):
        pipeline.transformer.blocks[i] = torch.compile(pipeline.transformer.blocks[i])
    for i in range(len(pipeline.transformer_2.blocks)):
        pipeline.transformer_2.blocks[i] = torch.compile(pipeline.transformer_2.blocks[i])
    print("Add Compile")

if GPU_memory_mode == "sequential_cpu_offload":
    replace_parameters_by_name(transformer, ["modulation",], device=device)
    replace_parameters_by_name(transformer_2, ["modulation",], device=device)
    transformer.freqs = transformer.freqs.to(device=device)
    transformer_2.freqs = transformer_2.freqs.to(device=device)
    pipeline.enable_sequential_cpu_offload(device=device)
elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
    convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
    convert_model_weight_to_float8(transformer_2, exclude_module_name=["modulation",], device=device)
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    convert_weight_dtype_wrapper(transformer_2, weight_dtype)
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_cpu_offload":
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_full_load_and_qfloat8":
    convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
    convert_model_weight_to_float8(transformer_2, exclude_module_name=["modulation",], device=device)
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    convert_weight_dtype_wrapper(transformer_2, weight_dtype)
    pipeline.to(device=device)
else:
    pipeline.to(device=device)

coefficients = get_teacache_coefficients(model_name) if enable_teacache else None
if coefficients is not None:
    print(f"Enable TeaCache with threshold {teacache_threshold} and skip the first {num_skip_start_steps} steps.")
    pipeline.transformer.enable_teacache(
        coefficients, num_inference_steps, teacache_threshold, num_skip_start_steps=num_skip_start_steps, offload=teacache_offload
    )
    pipeline.transformer_2.share_teacache(transformer=pipeline.transformer)

if cfg_skip_ratio is not None:
    print(f"Enable cfg_skip_ratio {cfg_skip_ratio}.")
    pipeline.transformer.enable_cfg_skip(cfg_skip_ratio, num_inference_steps)
    pipeline.transformer_2.share_cfg_skip(transformer=pipeline.transformer)

generator = torch.Generator(device=device).manual_seed(seed)

if lora_path is not None:
    pipeline = merge_lora(pipeline, lora_path, lora_weight, device=device)
    pipeline = merge_lora(pipeline, lora_high_path, lora_high_weight, device=device, sub_transformer_name="transformer_2")

with torch.no_grad():
    video_length = int((video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
    latent_frames = (video_length - 1) // vae.config.temporal_compression_ratio + 1

    if enable_riflex:
        pipeline.transformer.enable_riflex(k = riflex_k, L_test = latent_frames)
        pipeline.transformer_2.enable_riflex(k = riflex_k, L_test = latent_frames)

    input_video, input_video_mask, clip_image = get_image_to_video_latent(validation_image_start, None, video_length=video_length, sample_size=sample_size)

    sample = pipeline(
        prompt, 
        num_frames = video_length,
        negative_prompt = negative_prompt,
        height      = sample_size[0],
        width       = sample_size[1],
        generator   = generator,
        guidance_scale = guidance_scale,
        num_inference_steps = num_inference_steps,
        boundary = boundary,

        video      = input_video,
        mask_video   = input_video_mask,
        shift = shift,
    ).videos

if lora_path is not None:
    pipeline = unmerge_lora(pipeline, lora_path, lora_weight, device=device)
    pipeline = unmerge_lora(pipeline, lora_high_path, lora_high_weight, device=device, sub_transformer_name="transformer_2")

def save_results():
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    index = len([path for path in os.listdir(save_path)]) + 1
    prefix = str(index).zfill(8)
    if video_length == 1:
        video_path = os.path.join(save_path, prefix + ".png")

        image = sample[0, :, 0]
        image = image.transpose(0, 1).transpose(1, 2)
        image = (image * 255).numpy().astype(np.uint8)
        image = Image.fromarray(image)
        image.save(video_path)
    else:
        video_path = os.path.join(save_path, prefix + ".mp4")
        save_videos_grid(sample, video_path, fps=fps)

if ulysses_degree * ring_degree > 1:
    import torch.distributed as dist
    if dist.get_rank() == 0:
        save_results()
else:
    save_results()