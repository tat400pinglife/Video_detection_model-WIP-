import os
import tarfile
import urllib.request
import sys
from pathlib import Path
import time

# --- IMPORT YOUR GPU PROCESSOR ---
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))
from imports.gpu_proccesor import process_video_gpu

def download_archive(url, filename="Real_part_aa.tar.gz"):
    """Downloads the massive dataset archive into the imported_data folder."""
    root_dir = Path(__file__).resolve().parent.parent.parent
    imported_folder = root_dir / "data" / "imported_data"
    imported_folder.mkdir(parents=True, exist_ok=True)
    
    dest_path = imported_folder / filename
    
    # If the 30GB file is already there, don't redownload it!
    if dest_path.exists():
        print(f"Archive already exists at {dest_path}. Skipping download.")
        return dest_path
        
    print(f"Downloading 30GB archive to {dest_path}...")
    print("(This will take a while, make sure the server is plugged in!)")
    
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
        chunk_size = 1024 * 1024 * 10  # Download in 10 MB chunks
        downloaded = 0
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            print(f"Downloaded {downloaded / (1024*1024):.1f} MB...", end="\r")
            
    print("\nDownload complete!")
    return dest_path

def process_local_archive(archive_name="Real_part_aa.tar.gz", 
                          label="real", 
                          start_idx=50, 
                          end_idx=200, 
                          delete_after=True, 
                          tensorize_fn=process_video_gpu):
    """Iterates through the local archive, extracts a specific range, and processes them."""
    root_dir = Path(__file__).resolve().parent.parent.parent
    archive_path = root_dir / "data" / "imported_data" / archive_name
    videos_folder = root_dir / "data" / "videos" / label
    processed_folder = root_dir / "data" / "processed_data" / label
    
    if not archive_path.exists():
        print(f"Error: Archive not found at {archive_path}. Download it first!")
        return
        
    videos_folder.mkdir(parents=True, exist_ok=True)
    processed_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOpening local archive {archive_path}...")
    print(f"Targeting videos from index {start_idx} to {end_idx - 1}...")
    
    # mode 'r:gz' is specifically for reading local gzip-compressed tar files
    with tarfile.open(archive_path, mode='r:gz') as tar:
        current_idx = 0
        
        for member in tar:
            if member.isfile() and member.name.endswith('.mp4'):
                
                # Only process if the video falls within our targeted batch
                if start_idx <= current_idx < end_idx:
                    filename = os.path.basename(member.name)
                    temp_mp4_path = videos_folder / filename
                    
                    # 1. EXTRACT to data/videos/real
                    with tar.extractfile(member) as source, open(temp_mp4_path, 'wb') as target:
                        target.write(source.read())
                        
                        # deal with race conditions by ensuring data is fully written to disk before processing
                        target.flush()             # Force Python to send data to the OS buffer
                        os.fsync(target.fileno())  # Ensure OS buffer is flushed to disk

                    
                    print(f"\n[{current_idx}] Extracted {filename} -> Tensorizing...")
                    
                    # 2. PROCESS to data/processed_data/real
                    success = False
                    try:
                        numeric_label = 0.0 if label == "real" else 1.0
                        for attempt in range(3):
                            success = tensorize_fn(str(temp_mp4_path), output_dir=str(processed_folder), label=numeric_label)
                            if success:
                                break # It worked! Break out of the retry loop.
                                
                            print(f"  Attempt {attempt + 1} failed. Waiting 0.5s for OS to catch up...")
                            time.sleep(0.5)
                        if success:
                            print(f"Successfully saved tensor!")
                        else:
                            print(f"Tensorizer rejected {filename} (Silently returned False)")
                    except Exception as e:
                        print(f"⚠️ Error processing {filename}: {e}")
                        
                    # 3. CLEANUP
                    if delete_after and success and temp_mp4_path.exists():
                        os.remove(temp_mp4_path)
                    elif not success:
                        print(f"🛑 Kept {filename} in the videos folder for inspection.")

                        
                current_idx += 1
                
                # Stop iterating completely once we hit the end of our target range
                if current_idx >= end_idx:
                    print(f"\n✅ Finished processing batch [{start_idx}, {end_idx}).")
                    break

if __name__ == "__main__":
    REAL_URL = "https://modelscope.cn/datasets/cccnju/GenVideo-100K/resolve/master/Real_part_aa"
    
    run_download = False  # Set to True to download the archive
    run_processing = True  # Set to True to process a batch of videos from the archive


    root_dir = Path(__file__).resolve().parent.parent.parent
    print(root_dir)
    # Step 1: Secure the 30GB archive locally
    if(run_download):  # Set to False if you already have the archive downloaded
        download_archive(url=REAL_URL, filename="Real_part_aa.tar.gz")
    
    # Step 2: Extract and tensorize a specific chunk safely
    if(run_processing):  # Set to False if you just want to download without processing
        process_local_archive(
            archive_name="Real_part_aa.tar.gz",
            label="real",
            start_idx=1,      # Start at video 50
            end_idx=501,     # Stop before video 200
            delete_after=True, # Keeps your storage entirely clean
            tensorize_fn=process_video_gpu
        )
    print(f"Code was run, {('ran download') if run_download else 'skipped downloading download'}, {('ran processing') if run_processing else 'skipped processing'}")