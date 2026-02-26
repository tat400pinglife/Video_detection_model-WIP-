import os
import sys
import time
import torch
from pathlib import Path

# This is used to test the script running the scraper and then processing the same video with both pipelines to compare outputs.
# TODO: Update scrapper to try using gpu function to make the tensors.

# Add scraper to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts', 'scraper'))

from tiktok_process import save_video_batch
from imports.gpu_proccesor import compute_features_gpu
from imports.space import get_frames, extract_audio_spectrogram, compute_features, compress_features
from myStuff.compare_tensors import compare_pt_files

def run_apples_to_apples_test(video_path, output_dir, label):
    """Extracts frames ONCE so both CPU and GPU get the exact same random clip."""
    vid_path_obj = Path(video_path)
    cpu_path = Path(output_dir) / f"{vid_path_obj.stem}_CPU.pt"
    gpu_path = Path(output_dir) / f"{vid_path_obj.stem}_GPU.pt"
    
    print(f"\n--- Processing: {vid_path_obj.name} ---")
    
    # 1. Extract Frames & Audio ONCE (Shared by both)
    result = get_frames(str(video_path), num_frames=32)
    if result is None: return False
    frames_np, start_time, clip_duration = result
    audio_np = extract_audio_spectrogram(str(video_path), start_time, clip_duration)
    
    # 2. CPU RUN
    print("[CPU] Running space.py math...")
    feats_cpu = compute_features(frames_np, str(video_path), start_time, clip_duration, device=torch.device("cpu"))
    comp_cpu = compress_features(feats_cpu)
    comp_cpu['label'] = float(label)
    torch.save(comp_cpu, cpu_path)
    
    # 3. GPU RUN
    print("[GPU] Running gpu_proccesor.py math...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        feats_gpu = compute_features_gpu(frames_np, device)
        audio_t = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0).float()
        
    comp_gpu = {
        'rgb_batch': (feats_gpu['rgb_batch'] * 255).clamp(0, 255).to(torch.uint8).cpu(),
        'prnu':      feats_gpu['prnu'].squeeze(0).to(torch.float16).cpu(),
        'fft':       feats_gpu['fft'].squeeze(0).to(torch.float16).cpu(),
        'audio':     audio_t.squeeze(0).to(torch.float16).cpu(),
        'label':     float(label) 
    }
    torch.save(comp_gpu, gpu_path)
    
    # 4. COMPARE
    compare_pt_files(cpu_path, gpu_path)
    return True

if __name__ == "__main__":
    CSV_DIR = Path("./scripts/scraper/csvs")
    TEST_OUTPUT = Path("./myStuff/hold")
    TEMP_DIR = Path("./data/videos/temp") 
    
    TEST_OUTPUT.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    
    save_video_batch(
        link_folder=CSV_DIR,
        path=TEMP_DIR,        
        start=0,
        goal=1,               
        wait=2,               
        fn=run_apples_to_apples_test, 
        delete_after=True,    
        output_dir=str(TEST_OUTPUT),
        label=1.0 
    )