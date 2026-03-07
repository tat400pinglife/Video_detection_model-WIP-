import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import warnings

# Dont waste time and just use already made functions for frame/audio extraction. 
# We just want to replace the math-heavy compute_features() function with a CUDA-accelerated version.

from imports.space import get_frames, extract_audio_spectrogram

warnings.filterwarnings("ignore")

def compute_features_gpu(frames_numpy, device=torch.device('cuda')):
    """
    CUDA-accelerated feature extraction.
    Replaces sequential CPU loops with parallel PyTorch tensor math.
    """
    # 1. GPU Transfer: Move raw frames to VRAM immediately
    t_frames = torch.from_numpy(frames_numpy).float().to(device) / 255.0
    t_frames = t_frames.permute(0, 3, 1, 2) 
    
    T = t_frames.shape[0]
    mid_idx = T // 2

    # ==========================================================
    # A. Grayscale Conversion (Parallel Matrix Math)
    # ==========================================================
    weights = torch.tensor([0.299, 0.587, 0.114], device=device).view(1, 3, 1, 1)
    gray = (t_frames * weights).sum(dim=1, keepdim=True) 
    
    # ==========================================================
    # B. Temporal Motion (Vector Subtraction)
    # ==========================================================
    diff_seq = torch.abs(gray[1:] - gray[:-1]) 
    diff_mid = torch.abs(gray[mid_idx] - gray[mid_idx + 1] if mid_idx + 1 < T else gray[mid_idx] - gray[mid_idx-1])
    
    # ==========================================================
    # C. Frequency Analysis (Hardware FFT)
    # ==========================================================
    gray_mid = gray[mid_idx, 0]
    fft_complex = torch.fft.fft2(gray_mid)
    fft_shift = torch.fft.fftshift(fft_complex)
    mag = torch.log1p(torch.abs(fft_shift))
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-6)
    
    # ==========================================================
    # D. PRNU / Noise Fingerprint (Hardware Convolution)
    # try to mimic cv2.GaussianBlur(..., sigma=0)
    # ==========================================================
    # 1. Isolate only the middle frame (Matches space.py)
    g = gray[mid_idx].unsqueeze(0) # Shape: [1, 1, 256, 256]
    
    # 2. Hardcode OpenCV's secret binomial approximation kernel
    k1d = torch.tensor([1, 4, 6, 4, 1], dtype=torch.float32, device=device) / 16.0
    k2d = torch.outer(k1d, k1d).view(1, 1, 5, 5)
    
    # 3. Apply mirror padding to match cv2 BORDER_REFLECT_101
    g_pad = F.pad(g, (2, 2, 2, 2), mode='reflect')
    
    # 4. Blur and subtract to find the noise map
    blurred = F.conv2d(g_pad, k2d)
    prnu_map = g - blurred

    # Return intermediate tensors (Memory compression happens in process_video_gpu)
    return {
        "rgb_mid": t_frames[mid_idx].unsqueeze(0),
        "rgb_batch": t_frames,
        "diff_seq": diff_seq.unsqueeze(0), 
        "diff": diff_mid.unsqueeze(0).unsqueeze(0),
        "prnu": prnu_map,                        # Output: [1, 1, 256, 256]
        "fft": mag.unsqueeze(0).unsqueeze(0)     # Output: [1, 1, 256, 256]
    }

def process_video_gpu(video_path, output_dir, label, max_frames=32):
    """
    Main Execution Hook.
    Integrates frame extraction, CUDA math, and memory compression.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vid_path_obj = Path(video_path)
    out_path = Path(output_dir) / f"{vid_path_obj.stem}.pt"\

    try:
        # 1. CPU Extraction (Using space.py's methods for exact parity)
        result = get_frames(str(video_path), num_frames=max_frames)
        if result is None: return False
        
        frames_np, start_time, clip_duration = result
        audio_np = extract_audio_spectrogram(str(video_path), start_time, clip_duration)
        
        # 2. CUDA Hardware Math
        with torch.no_grad():
            features = compute_features_gpu(frames_np, device)
            audio_t = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0).float()

        # 3. Memory Compression
        # Drops precision to uint8/float16 to match space.py disk footprint
        data_to_save = {
            'rgb_batch': (features['rgb_batch'] * 255).clamp(0, 255).to(torch.uint8).cpu(),
            'prnu':      features['prnu'].squeeze(0).to(torch.float16).cpu(),
            'fft':       features['fft'].squeeze(0).to(torch.float16).cpu(),
            'audio':     audio_t.squeeze(0).to(torch.float16).cpu(),
        }
        
        # This fixed something or was a false alarm idk
        data_to_save['label'] = float(label)
        
        torch.save(data_to_save, out_path)
        return True

    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return False