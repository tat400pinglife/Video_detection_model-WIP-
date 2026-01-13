import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings

# Import your extraction functions
from utils import get_frames, compute_features, extract_audio_spectrogram

# Suppress librosa warnings about short audio
warnings.filterwarnings("ignore")

def process_dataset(input_dir, output_dir, label, max_videos=None):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    videos = list(input_path.rglob("*.mp4")) + list(input_path.rglob("*.avi"))
    print(f"Found {len(videos)} videos in {input_dir}")
    
    if max_videos: videos = videos[:max_videos]
    
    success_count = 0
    
    for vid in tqdm(videos):
        try:
            # 1. Extract Frames (Visual)
            frames = get_frames(vid)
            if frames is None: continue
            
            # 2. Compute Visual Features (RGB, PRNU)
            # We pass device='cpu' because we want to save them as CPU tensors
            feats = compute_features(frames, vid, device=torch.device("cpu"))
            
            # 3. Extract Audio (Spectrogram)
            # Note: compute_features now handles this internally if you updated utils.py
            # But just to be safe/explicit, we can pull it from the feats dict
            # or calculate it manually if compute_features doesn't do it yet.
            
            # Using the feats dict from your updated utils.py:
            spectrogram = feats['audio'] # This is already a Tensor (1, 1, 128, 128)
            
            # If you are using the OLD utils.py that doesn't return 'audio' in feats:
            # spec_numpy = extract_audio_spectrogram(vid)
            # spectrogram = torch.from_numpy(spec_numpy).unsqueeze(0).float()

            # 4. Save Everything to One File
            save_name = vid.stem + ".pt"
            
            torch.save({
                'rgb': feats['rgb_batch'].clone(), # (32, 3, 256, 256)
                'prnu': feats['prnu'].clone(),     # (1, 1, 256, 256)
                'audio': spectrogram.clone(),      # (1, 1, 128, 128)
                'label': float(label)
            }, output_path / save_name)
            
            success_count += 1
            
        except Exception as e:
            print(f"Failed {vid.name}: {e}")
            continue
            
    print(f"Successfully processed {success_count} videos.")

if __name__ == "__main__":
    # Adjust paths to your actual folders
    print("Processing REAL videos...")
    process_dataset("data/videos/real", "data/processed_data/real", label=0.0)
    
    print("Processing FAKE videos...")
    process_dataset("data/videos/fake", "data/processed_data/fake", label=1.0)