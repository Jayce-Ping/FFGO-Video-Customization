import os
import re
import csv
from typing import List, Optional
import openai
from PIL import Image
import imageio.v3 as iio
import base64
from io import BytesIO
from tqdm import tqdm

os.environ['FPS_MAX_FRAMES'] = '16'

MAGIC_PREFIX = """ad23r2 the camera view suddenly changes."""

PROMPT_TEMPLATE = """
### Task Description

You are given a video and several images. Generate a descriptive caption for the video that prominently features the components shown in the images. Wrap your final text in <caption>. . . </caption> tags.
The caption must highlight the significance and role of these components throughout the video, while omitting filler such as "The scene unfolds with a whimsical and heartwarming narrative, emphasizing the simple joys of life through the Teddy Bear's endearing actions".

### Examples of Descriptive Captions

1. Film quality, professional quality, rich details. The video begins to show the surface of a pond, and the camera slowly zooms in to a close-up. The water surface begins to bubble, and then a blonde woman is seen coming out of the lotus pond soaked all over, showing the subtle changes in her facial expression.

2. A professional male diver performs an elegant diving maneuver from a high platform. Full-body side view captures him wearing bright red swim trunks in an upside-down posture with arms fully extended and legs straight and pressed together. The camera pans downward as he dives into the water below.
"""


def load_video_frames(video_path: str, fps: Optional[int] = None, max_frames: int = 16) -> List[Image.Image]:
    """
    Args:
        - video_path: Path to the video file.
        - fps: Desired frames per second to sample from the video. If None, use original fps.
        - max_frames: Maximum number of frames to return.
    """
    frames = [Image.fromarray(frame) for frame in iio.imread(video_path)]
    
    if fps is not None:
        metadata = iio.immeta(video_path)
        original_fps = metadata.get('fps', 30)
        step = original_fps / fps
        indices = [int(i * step) for i in range(int(len(frames) / step))]
        frames = [frames[i] for i in indices if i < len(frames)]
    
    if len(frames) > max_frames:
        indices = torch.linspace(0, len(frames) - 1, max_frames).round().long()
        frames = [frames[i] for i in indices]
    
    return frames


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()


def extract_caption(text: str) -> str:
    """Extract text between <caption> tags."""
    match = re.search(r'<caption>(.*?)</caption>', text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def generate_video_caption(
    client: openai.OpenAI,
    video_path: str,
    component_images: List[Image.Image],
    model: str = "Qwen3-VL-30B-A3B-Instruct",
    total_pixels: int = 24576 * 32 * 32,
    max_tokens: int = 1024,
    temperature : float = 0.7,
) -> str:
    """Generate caption using video_url (推荐方式)."""
    
    content = [
        {"type": "text", "text": PROMPT_TEMPLATE},
        {
            "type": "video_url",
            "video_url": {"url": f"file://{os.path.abspath(video_path)}"}
        }
    ]
    
    for img in component_images:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_to_base64(img)}"}
        })
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body={
            "mm_processor_kwargs": {
                "total_pixels": total_pixels,
                "do_sample_frames": True
            }
        }
    )
    
    return extract_caption(response.choices[0].message.content)


def main():
    client = openai.OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1"
    )
    model = 'Qwen3-VL-235B-A22B-Instruct'
    
    dataset_dir = os.path.expanduser('~/jcy/datasets/tryon_dataset/')
    dataset_dir = os.path.realpath(dataset_dir)
    video_dir = os.path.join(dataset_dir, 'videos')
    image_dir = os.path.join(dataset_dir, 'images')
    output_file = os.path.join(dataset_dir, 'video_captions.csv')
    
    # Group images by video ID
    video_images = {}
    for img_file in os.listdir(image_dir):
        if img_file.endswith(('.jpg', '.png')):
            # Extract video ID from image filename (assumes format: videoID_*.png)
            video_id = os.path.splitext(img_file)[0]
            if video_id not in video_images:
                video_images[video_id] = []
            video_images[video_id].append(img_file)
    
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'image_path', 'prompt', 'note'])
        
        for i, video_id in enumerate(tqdm(sorted(video_images.keys()), desc="Processing Videos")):
            video_path = os.path.join(video_dir, f'{video_id}.mp4')
            if not os.path.exists(video_path):
                continue
            
            # Load component images
            component_images = [
                Image.open(os.path.join(image_dir, img_file))
                for img_file in sorted(video_images[video_id])
            ]
            
            caption = generate_video_caption(
                client=client,
                video_path=video_path,
                component_images=component_images,
                model=model,
            )

            caption = MAGIC_PREFIX + " " + caption
            
            # Write one row per image with same caption
            for idx, img_file in enumerate(sorted(video_images[video_id])):
                img_path = os.path.join('images', img_file)
                writer.writerow([i, img_path, caption, ''])


if __name__ == "__main__":
    main()