import os
from pathlib import Path
from imports.space import process_dataset
from imports.gpu_proccesor import process_video_gpu

USE_GPU = False 

def process_dataset_gpu(input_dir, output_dir, label):
    """
    Helper function to batch process a directory using the new GPU tensorizer.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    videos = list(input_path.rglob("*.mp4")) + list(input_path.rglob("*.avi")) + list(input_path.rglob("*.mov"))
    print(f"Found {len(videos)} videos in {input_dir}")
    
    success_count = 0
    for vid in videos:
        print(f" [GPU] Processing: {vid.name}")
        # Call the GPU pipeline we built
        success = process_video_gpu(str(vid), str(output_path), label=label)
        if success:
            success_count += 1
            
    print(f"Successfully processed {success_count}/{len(videos)} videos using CUDA.")

if __name__ == "__main__":
    REAL_VIDEO_DIR = "./data/videos/real"
    FAKE_VIDEO_DIR = "./data/videos/fake"
    
    print(f"Starting Tensor Creation (USE_GPU = {USE_GPU})")
    
    print(f"\nProcessing REAL videos from: {REAL_VIDEO_DIR}")
    if USE_GPU:
        process_dataset_gpu(REAL_VIDEO_DIR, "./data/processed_data/real", label=0.0)
    else:
        process_dataset(REAL_VIDEO_DIR, "./data/processed_data/real", label=0.0)
    
    print(f"\nProcessing FAKE videos from: {FAKE_VIDEO_DIR}")
    if USE_GPU:
        process_dataset_gpu(FAKE_VIDEO_DIR, "./data/processed_data/fake", label=1.0)
    else:
        process_dataset(FAKE_VIDEO_DIR, "./data/processed_data/fake", label=1.0)
    
    print("\n--- Processing Complete ---")