import sys
from pathlib import Path
from imports.gpu_proccesor import process_video_gpu

sys.path.append(str(Path(__file__).resolve().parent / 'scripts' / 'scraper'))
from tiktok_process import save_video_batch

if __name__ == "__main__":
    print("scrape plz")
    
    save_video_batch(
        link_folder=Path("./scripts/scraper/csvs"),
        path=Path("./data/videos/temp"),
        start=0,
        goal=9000,                  # big number
        wait=3,                       # wait 3 seconds to avoid ban.
        fn=process_video_gpu,         # GPU please work
        delete_after=True,            # Deletes the mp4 after processing
        output_dir="./data/processed_data/fake",
        label=1.0 
    )