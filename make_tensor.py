import time
from pathlib import Path
from tqdm import tqdm

# Import the main processing hook from your merged script
from imports.combined import process_video_gpu

def process_dataset(input_dir, output_dir, max_frames=32, num_clips=3):
    print("=== Starting Batch Video Processing ===")
    
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    
    # 1. Verification & Setup
    if not in_path.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return
        
    # 2. Find all videos recursively
    extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
    videos = []
    for ext in extensions:
        videos.extend(list(in_path.rglob(ext)))
        
    if not videos:
        print(f"No videos found in '{input_dir}' or its subfolders.")
        return
        
    print(f"Found {len(videos)} videos.")
    print(f"Saving tensors to: {out_path.resolve()}")
    print("-" * 50)
    
    success_count = 0
    failed_videos = []
    start_time = time.time()
    
    # 3. Processing Loop
    for vid_path in tqdm(videos, desc="Processing Videos", unit="vid"):
        
        # ==========================================================
        # LABEL & ROUTING LOGIC
        # Determines label AND the output subfolder name
        # ==========================================================
        parent_folder = vid_path.parent.name.lower()
        
        if "real" in parent_folder or "original" in parent_folder:
            label = 0.0
            target_subfolder = "real"
        elif "fake" in parent_folder or "manipulated" in parent_folder:
            label = 1.0
            target_subfolder = "fake"
        else:
            label = -1.0 
            target_subfolder = "uncategorized"
            
        # Create the specific subfolder path (e.g., ./processed_tensors/fake/)
        specific_out_dir = out_path / target_subfolder
        specific_out_dir.mkdir(parents=True, exist_ok=True)
            
        # 4. Run the GPU Processor
        success = process_video_gpu(
            video_path=str(vid_path),
            output_dir=str(specific_out_dir), # Route to the specific subfolder
            label=label,
            max_frames=max_frames,
            num_clips=num_clips
        )
        
        if success:
            success_count += 1
        else:
            failed_videos.append(vid_path.name)
            
    # 5. Summary Statistics
    total_time = time.time() - start_time
    print("\n" + "=" * 50)
    print(f"Batch Processing Complete in {total_time:.2f} seconds.")
    print(f"Summary: {success_count}/{len(videos)} videos successfully processed.")
    
    if failed_videos:
        print(f"\nFailed to process {len(failed_videos)} videos.")
        for failed in failed_videos[:5]:
            print(f"  - {failed}")
        if len(failed_videos) > 5:
            print(f"  - ... and {len(failed_videos) - 5} more. Check file integrity.")

if __name__ == "__main__":
    
    INPUT_DIRECTORY = "./Video Folder"        # The root folder containing all subfolders/videos
    OUTPUT_DIRECTORY = "./data/processed_data" # The root folder for tensors
    
    process_dataset(
        input_dir=INPUT_DIRECTORY,
        output_dir=OUTPUT_DIRECTORY,
        max_frames=32,  
        num_clips=3     
    )