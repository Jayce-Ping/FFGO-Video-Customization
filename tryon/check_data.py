import os
import cv2
import glob
from tqdm import tqdm

# Set this to your dataset folder
DATASET_ROOT = '/home/users/astar/cfar/stuchengyou/jcy/datasets/tryon_dataset/'

def check_video(file_path):
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return False, "Cannot open file"
        
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        
        cap.release()

        if width <= 0 or height <= 0:
            return False, f"Invalid dimensions: {width}x{height}"
        
        if frame_count == 0:
            return False, "Zero frames"
            
        return True, "OK"
    except Exception as e:
        return False, str(e)

print(f"Scanning {DATASET_ROOT}...")
video_files = glob.glob(os.path.join(DATASET_ROOT, "*.mp4"))

bad_files = []

for video_path in tqdm(video_files):
    is_valid, reason = check_video(video_path)
    if not is_valid:
        print(f"\n[BAD FILE] {video_path}")
        print(f"Reason: {reason}")
        bad_files.append(video_path)

print(f"\nScan complete. Found {len(bad_files)} bad files.")
if len(bad_files) > 0:
    print("Please remove these files from your dataset metadata.")