import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings

# Import from your utils file
from imports.utils import get_frames, compute_features 

warnings.filterwarnings("ignore")

def process_dataset(input_dir, output_dir, label, max_videos=None):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    videos = list(input_path.rglob("*.mp4")) + list(input_path.rglob("*.avi")) + list(input_path.rglob("*.mov"))
    print(f"Found {len(videos)} videos in {input_dir}")
    
    if max_videos: videos = videos[:max_videos]
    
    success_count = 0
    
    for vid in tqdm(videos):
        try:
            # 1. Extract Frames
            frames = get_frames(str(vid))
            if frames is None: continue
            
            # 2. Compute Features 
            feats = compute_features(frames, str(vid), device=torch.device("cpu"))
            
            # 3. Save to Disk
            save_name = vid.stem + ".pt"
            
            torch.save({
                'rgb_mid': feats['rgb_mid'].squeeze(0).clone(), # [3, 256, 256]
                'prnu':    feats['prnu'].squeeze(0).clone(),    # [1, 256, 256]
                'fft':     feats['fft'].squeeze(0).clone(),     # [1, 256, 256]
                'diff':    feats['diff'].squeeze(0).clone(),    # [1, 256, 256]
                'audio':   feats['audio'].squeeze(0).clone(),   # [1, 128, 128]
                'label':   float(label)
            }, output_path / save_name)
            
            success_count += 1
            
        except Exception as e:
            # print(f"Failed {vid.name}: {e}")
            continue
            
    print(f"Successfully processed {success_count} videos.")

if __name__ == "__main__":
    # Adjust paths
    print("Processing REAL videos...")
    process_dataset("./data/videos/real", "./data/processed_data/real", label=0.0)
    
    print("Processing FAKE videos...")
    process_dataset("./data/videos/fake", "./data/processed_data/fake", label=1.0)