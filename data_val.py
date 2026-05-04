import subprocess
from pathlib import Path
from tqdm import tqdm

def get_video_codec(filepath):
    """Uses ffprobe to determine the video codec of a file."""
    cmd = [
        'ffprobe', 
        '-v', 'error', 
        '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name',
        '-of', 'default=noprint_wrappers=1:nokey=1', 
        str(filepath)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip().lower()
    except subprocess.CalledProcessError:
        return None
    except FileNotFoundError:
        print("❌ Error: 'ffprobe' is not installed or not in your system PATH.")
        exit(1)

def transcode_to_h264(input_path):
    """Transcodes a video to H.264, replacing the original file."""
    # Create a temporary output filename
    temp_output = input_path.with_name(f"{input_path.stem}_temp_h264{input_path.suffix}")
    
    cmd = [
        'ffmpeg', '-y', 
        '-i', str(input_path),
        '-c:v', 'libx264', 
        '-preset', 'fast', 
        '-crf', '23',        # 23 is a great balance of quality and file size
        '-c:a', 'copy',      # Don't touch the audio, just copy it
        str(temp_output)
    ]
    
    try:
        # Run ffmpeg quietly
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        # If successful, overwrite the original file with the new H.264 version
        temp_output.replace(input_path)
        return True
        
    except subprocess.CalledProcessError:
        # If it fails, clean up the broken temporary file
        if temp_output.exists():
            temp_output.unlink()
        return False
    except FileNotFoundError:
        print("❌ Error: 'ffmpeg' is not installed or not in your system PATH.")
        exit(1)

def sanitize_dataset(dataset_dir):
    print("=== AV1 to H.264 Dataset Sanitizer ===")
    target_dir = Path(dataset_dir)
    
    if not target_dir.exists():
        print(f"❌ Error: Directory '{dataset_dir}' not found.")
        return

    # 1. Find all potential video files
    extensions = ["*.mp4", "*.mkv", "*.webm", "*.avi", "*.mov"]
    videos = []
    for ext in extensions:
        videos.extend(list(target_dir.rglob(ext)))

    if not videos:
        print("⚠️ No videos found to check.")
        return

    print(f"🔍 Scanning {len(videos)} videos for AV1 codecs...\n")
    
    av1_files = []
    
    # 2. Identify AV1 files
    for vid in tqdm(videos, desc="Checking Codecs", unit="vid"):
        codec = get_video_codec(vid)
        if codec == 'av1':
            av1_files.append(vid)

    if not av1_files:
        print("\n✅ Dataset is clean! No AV1 files found. You are good to go.")
        return

    print(f"\n⚠️ Found {len(av1_files)} AV1 videos. Starting conversion...")

    # 3. Transcode the flagged files
    success_count = 0
    failed_files = []
    
    for vid in tqdm(av1_files, desc="Converting to H.264", unit="vid"):
        if transcode_to_h264(vid):
            success_count += 1
        else:
            failed_files.append(vid.name)

    # 4. Summary
    print("\n" + "=" * 50)
    print(f"✅ Conversion Complete: {success_count}/{len(av1_files)} files fixed.")
    
    if failed_files:
        print(f"❌ Failed to convert {len(failed_files)} files:")
        for f in failed_files[:5]:
            print(f"  - {f}")
        if len(failed_files) > 5:
            print(f"  - ...and {len(failed_files) - 5} more.")

if __name__ == "__main__":
    # Point this to your main dataset folder
    DATASET_DIRECTORY = "./Video Folder" 
    
    sanitize_dataset(DATASET_DIRECTORY)