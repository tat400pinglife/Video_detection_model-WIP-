import os
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# TODO: eventually will need to parallelize this since there is 500k videos, or 6mil dont shoot the messenger
#       also will need to convert to tensors so the model can actually process them
#       also will need to find out the normalized size of dataset, right now im just testing own videos
#       bouta eat smth maybe take a break


SOURCE_DIR = Path("./data/videos")
OUTPUT_DIR = Path("./data/frames")
TARGET_SIZE = 256
NUM_FRAMES = 32

def get_frames(video_path, size=224, num_frames=16):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        cap.release()
        return np.zeros((num_frames, size, size, 3), dtype=np.uint8)

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    current_idx = 0
    collected = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if current_idx in indices:
            frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            collected += 1
        current_idx += 1
        if collected >= num_frames: break
            
    cap.release()
    
    frames = np.array(frames)
    if len(frames) < num_frames:
        padding = np.zeros((num_frames - len(frames), size, size, 3), dtype=np.uint8)
        if len(frames) > 0: frames = np.concatenate([frames, padding], axis=0)
        else: frames = padding
            
    return frames

def compute_fft(frames):
    gray = np.dot(frames[..., :3], [0.299, 0.587, 0.114])
    ffts = []
    for i in range(len(gray)):
        f = np.fft.fft2(gray[i])
        fshift = np.fft.fftshift(f)
        magnitude = np.log1p(np.abs(fshift))
        m_min, m_max = magnitude.min(), magnitude.max()
        if m_max - m_min > 1e-5:
            magnitude = (magnitude - m_min) / (m_max - m_min)
        else:
            magnitude = np.zeros_like(magnitude)
        ffts.append(magnitude)
    return np.array(ffts, dtype=np.float32)

def compute_diff(frames):
    gray = np.dot(frames[..., :3], [0.299, 0.587, 0.114])
    diffs = [np.zeros_like(gray[0])]
    for i in range(1, len(gray)):
        d = np.abs(gray[i] - gray[i-1])
        diffs.append(d)
    return np.array(diffs, dtype=np.float32) / 255.0

def process_folder(folder_name):
    input_path = SOURCE_DIR / folder_name
    output_path = OUTPUT_DIR / folder_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    videos = list(input_path.glob("*.mp4"))
    print(f"Processing {len(videos)} videos in '{folder_name}'...")
    
    for video_file in tqdm(videos):
        save_path = output_path / f"{video_file.stem}.pt"
        
        # Extract & Compute
        rgb = get_frames(video_file, size=TARGET_SIZE, num_frames=NUM_FRAMES)
        fft = compute_fft(rgb)
        diff = compute_diff(rgb)
        
        rgb_norm = rgb.astype(np.float32) / 255.0

        data = {
            "rgb": torch.from_numpy(rgb_norm).permute(0, 3, 1, 2), # (T, C, H, W)
            "fft": torch.from_numpy(fft).unsqueeze(1),             # (T, 1, H, W)
            "diff": torch.from_numpy(diff).unsqueeze(1),           # (T, 1, H, W)
            "label": 1 if folder_name == "fake" else 0             # Embed label
        }
        
        torch.save(data, save_path)

if __name__ == "__main__":
    process_folder("real")
    process_folder("fake")
    print("Done! Check data/frames/")