import sys
import importlib
from pathlib import Path

# --- SETUP PATHS ---
root_dir = Path(__file__).resolve().parent
sys.path.append(str(root_dir / 'scripts' / 'scraper'))
sys.path.append(str(root_dir)) 

# --- IMPORT YOUR GPU PROCESSOR ---
from imports.gpu_proccesor import process_video_gpu

# Safely import the scraper even with the hyphen in the filename
try:
    yt_scraper = importlib.import_module("youtube-scraper")
except ModuleNotFoundError:
    # Fallback just in case someone renames it to 'youtube_scraper.py' later
    yt_scraper = importlib.import_module("youtube_scraper")

if __name__ == "__main__":
    print("Starting YouTube bulk download and tensorization...")
    
    # Run the exact same Producer-Consumer loop as the TikTok scraper
    yt_scraper.save_video_batch(
        # IMPORTANT: Make sure your youtube.csv is in this folder, separated from TikTok CSVs!
        link_folder=Path("./scripts/youtube/youtube_csvs"), 
        path=Path("./data/videos/temp"),
        start=0,
        goal=5,                     # The 350 new links you are targeting
        wait=5,                       # 5 second wait to avoid YouTube rate limits/bans
        fn=process_video_gpu,         # Feed the .mp4 straight to the GPU
        delete_after=True,            # Instantly delete the .mp4 to save space
        output_dir="./data/processed_data/real", 
        label=0.0                     # 1.0 = Fake, 0.0 = Real (needs to be real)
    )