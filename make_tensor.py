import os
from pathlib import Path
from imports.gpu_proccesor import VideoTensorizer  # Imports the new modular GPU class
# create_tensors.py
from imports.utils import process_dataset
RUNOLDUTILS = True
MAXFRAMES = 32


def process_directory(input_dir, output_dir, label, processor):
    """
    Scans a directory and processes videos using the GPU Tensorizer.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all videos
    videos = list(input_path.rglob("*.mp4")) + list(input_path.rglob("*.avi")) + list(input_path.rglob("*.mov"))
    print(f"\nFound {len(videos)} videos in {input_dir}")
    
    success_count = 0
    
    for vid in videos:
        save_name = vid.stem + ".pt"
        save_path = output_path / save_name
        
        # Skip if already done
        if save_path.exists():
            print(f"Skipping {vid.name} (Already processed)")
            continue
            
        # Use the GPU Processor
        # max_frames=None lets it process the FULL video as you requested (200GB storage space)
        success = processor.process_video(
            video_path=str(vid), 
            output_path=str(save_path), 
            label=label,
            max_frames=MAXFRAMES 
        )
        
        if success:
            success_count += 1

    print(f"Batch Complete. Processed {success_count}/{len(videos)} videos.")

if __name__ == "__main__":
    # 1. Initialize GPU Processor ONCE
    # This automatically runs the "cpu vs gpu" check inside gpu_processor.py
    processor = VideoTensorizer() 
    
    # 2. Define Directories
    REAL_VIDEO_DIR = "./data/videos/real"
    FAKE_VIDEO_DIR = "./data/videos/fake"
    
    REAL_OUTPUT = "./data/processed_data/real"
    FAKE_OUTPUT = "./data/processed_data/fake"
    
    print("Starting GPU-Accelerated Tensor Creation...")

    # video used https://www.youtube.com/watch?v=xwuyBTTuUrQ

    # 3. Run Batches
    if (RUNOLDUTILS):
    
        print("Starting (old) Tensor Creation")
    
        print(f"\nProcessing REAL videos from: {REAL_VIDEO_DIR}")
        process_dataset(REAL_VIDEO_DIR, "./data/processed_data/real", label=0.0)
    
        print(f"\nProcessing FAKE videos from: {FAKE_VIDEO_DIR}")
        process_dataset(FAKE_VIDEO_DIR, "./data/processed_data/fake", label=1.0)
    
        print("\n--- Processing Complete ---")
    else:
        if os.path.exists(REAL_VIDEO_DIR):
            process_directory(REAL_VIDEO_DIR, REAL_OUTPUT, label=0.0, processor=processor)
        else:
            print(f"Warning: {REAL_VIDEO_DIR} not found.")
        
        if os.path.exists(FAKE_VIDEO_DIR):
            process_directory(FAKE_VIDEO_DIR, FAKE_OUTPUT, label=1.0, processor=processor)
        else:
            print(f"Warning: {FAKE_VIDEO_DIR} not found.")