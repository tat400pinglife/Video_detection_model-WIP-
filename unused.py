import os
import yt_dlp
from pathlib import Path

# --- CONFIGURATION ---
DATA_ROOT = "./data/videos"
MAX_VIDEOS_PER_SOURCE = 50  # Limit per channel/query to ensure diversity
video_duration_limit = 30   # Download only first 30 seconds (Saves bandwidth!)

# SOURCES
# 1. REAL: We want high-quality "talking heads" (News, TED Talks, Interviews)
REAL_SOURCES = [
    "https://www.youtube.com/@TED",           # TED Talks (Perfect lighting/quality)
    "https://www.youtube.com/@BBCNews",       # News Anchors
    "https://www.youtube.com/@CNN",           # News Anchors
    "https://www.youtube.com/@Wired",         # Interviews
    "https://www.youtube.com/@Vogue",         # 73 Questions (Good face data)
]

# 2. FAKE: We want a mix of "High Quality" (Sora) and "Low Quality" (Apps)
FAKE_QUERIES = [
    "Sora AI video examples",
    "HeyGen AI avatar demo",
    "Synthesia AI example",
    "Deepfake celebrity impression",
    "AI generated news anchor",
    "Runway Gen-2 video examples",
    "Pika labs ai video",
    "Stable Video Diffusion examples"
]

def download_videos(sources, label, is_search=False):
    output_path = Path(DATA_ROOT) / label
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n--- Downloading {label.upper()} Dataset ---")
    
    for source in sources:
        print(f"Processing: {source}")
        
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': f'{output_path}/%(id)s.%(ext)s',
            'noplaylist': True,
            
            # LIMITS
            'max_downloads': MAX_VIDEOS_PER_SOURCE,
            'match_filter': yt_dlp.utils.match_filter_func("!is_live"), # No livestreams
            
            # TRIMMING (Crucial for speed)
            # Downloads only seconds 0 to 30
            'download_ranges': yt_dlp.utils.download_range_func(None, [(0, video_duration_limit)]),
            'force_keyframes_at_cuts': True,
            
            # QUIET MODE
            'quiet': True,
            'no_warnings': True,
            'progress_hooks': [lambda d: print(".", end="", flush=True) if d['status'] == 'finished' else None]
        }
        
        # If it's a search query, use the 'ytsearch' prefix
        url = f"ytsearch{MAX_VIDEOS_PER_SOURCE}:{source}" if is_search else source
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        except Exception as e:
            print(f"\nSkipping source {source} due to error: {e}")

if __name__ == "__main__":
    # 1. Download REAL Videos (From Channels)
    #download_videos(REAL_SOURCES, "real", is_search=False)
    
    # 2. Download FAKE Videos (From Search Queries)
    download_videos(FAKE_QUERIES, "fake", is_search=True)
    
    print("\n\n--- Download Complete ---")
    print(f"Check {DATA_ROOT} for your files.")