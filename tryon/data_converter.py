import csv
import json
import random
import os
from pathlib import Path
from typing import List, Dict

def split_csv_to_train_test(
    csv_path: str,
    train_json_path: str,
    test_csv_path: str,
    base_dir: str = "/workspace/Data/train",
    split_ratio: float = 0.5,
    seed: int = 42
):
    """
    Split CSV into train (JSON format) and test (CSV format) with 1:1 ratio.
    
    Args:
        csv_path: Input CSV file (video_captions.csv)
        train_json_path: Output train JSON path
        test_csv_path: Output test CSV path
        base_dir: Base directory prefix for JSON paths
        split_ratio: Train split ratio (default 0.5 for 1:1)
        seed: Random seed for reproducibility
    """
    # Read CSV data
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    
    # Shuffle with seed
    random.seed(seed)
    random.shuffle(data)
    
    # Split
    split_idx = int(len(data) * split_ratio)
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    # Convert train to JSON format
    train_json = []
    for row in train_data:
        image_path = row['image_path']
        video_path = str(Path(image_path).with_suffix('.mp4')).replace('images', 'videos')
        
        # Add base_dir prefix
        if base_dir:
            image_path_full = os.path.join(base_dir, image_path)
            video_path_full = os.path.join(base_dir, video_path)
        else:
            image_path_full = image_path
            video_path_full = video_path
        
        entry = {
            "file_path": video_path_full,
            "text": row['prompt'],
            "image_path": image_path_full,
            "type": "video"
        }
        train_json.append(entry)
    
    # Save train as JSON
    with open(train_json_path, 'w', encoding='utf-8') as f:
        json.dump(train_json, f, ensure_ascii=False, indent=2)
    
    # Save test as CSV (keep original format)
    with open(test_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'image_path', 'prompt', 'note'])
        
        for idx, row in enumerate(test_data):
            writer.writerow([idx, row['image_path'], row['prompt'], row.get('note', '')])
    
    print(f"Total: {len(data)} samples")
    print(f"Train: {len(train_json)} samples -> {train_json_path}")
    print(f"Test:  {len(test_data)} samples -> {test_csv_path}")


if __name__ == "__main__":
    base_dir = os.path.expanduser("~/jcy/datasets/tryon_dataset")
    
    split_csv_to_train_test(
        csv_path=f"{base_dir}/video_captions.csv",
        train_json_path=f"{base_dir}/train.json",
        test_csv_path=f"{base_dir}/test.csv",
        base_dir=base_dir,  # JSON中的路径前缀
        split_ratio=0.5,
        seed=42
    )