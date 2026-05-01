import cv2
import numpy as np
import torch
import torch.nn.functional as F
import librosa
import random
import warnings
import subprocess
from tqdm import tqdm
from pathlib import Path

warnings.filterwarnings("ignore")

# ==========================================================
# AUDIO EXTRACTION
# ==========================================================

def load_audio_from_video(video_path, start_time, duration, sr=16000):
    """Uses ffmpeg to extract an exact audio slice directly into a NumPy array."""
    command = [
        'ffmpeg',
        '-ss', str(start_time),      # Seek to start time
        '-i', video_path,            # Input video
        '-t', str(duration),         # Duration to extract
        '-f', 'f32le',               # Format: 32-bit float little-endian
        '-ac', '1',                  # Audio channels: 1 (Mono)
        '-ar', str(sr),              # Audio sample rate
        '-loglevel', 'quiet',        # Suppress ffmpeg console output
        '-'                          # Pipe output to stdout
    ]
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        raw_audio, _ = process.communicate()
        audio_array = np.frombuffer(raw_audio, dtype=np.float32)
        return audio_array, sr
    except Exception:
        return np.zeros(int(sr * duration), dtype=np.float32), sr

def extract_audio_spectrogram(video_path, start_time=0.0, duration=1.06, target_shape=(128, 128)):
    try:
        y, sr = load_audio_from_video(str(video_path), start_time, duration, sr=16000)
        
        if len(y) < 1000: return np.zeros(target_shape, dtype=np.float32)
        
        # Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=target_shape[0])
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize 0-1
        mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-6)
        
        # Resize to square for CNN compatibility
        mel_spec_resized = cv2.resize(mel_spec_db, (target_shape[1], target_shape[0]))
        
        return mel_spec_resized
    except Exception:
        return np.zeros(target_shape, dtype=np.float32)

# ==========================================================
# VIDEO FRAME EXTRACTION
# ==========================================================

def get_frames(video_path, size=256, num_frames=64):
    """Extracts a single random clip from the video."""
    path_obj = Path(video_path).resolve()
    if not path_obj.exists(): return None

    cap = cv2.VideoCapture(str(path_obj))
    if not cap.isOpened(): return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps): fps = 30.0 
    
    if total_frames <= num_frames:
        start_idx = 0
    else:
        max_start = total_frames - num_frames
        start_idx = random.randint(0, max_start)
        
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
    
    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret: break
        
        h, w = frame.shape[:2]
        start_y = max(0, h // 2 - size // 2)
        start_x = max(0, w // 2 - size // 2)
        
        frame = frame[start_y : start_y + size, start_x : start_x + size]
        
        if frame.shape[0] < size or frame.shape[1] < size:
            pad_y = max(0, size - frame.shape[0])
            pad_x = max(0, size - frame.shape[1])
            frame = cv2.copyMakeBorder(frame, 0, pad_y, 0, pad_x, cv2.BORDER_CONSTANT, value=[0,0,0])

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        
    cap.release()
    frames = np.array(frames)
    
    if len(frames) > 0 and len(frames) < num_frames:
        pad_len = num_frames - len(frames)
        last_frame = frames[-1:]
        padding = np.repeat(last_frame, pad_len, axis=0)
        frames = np.concatenate([frames, padding], axis=0)
        
    if len(frames) == 0: return None
        
    start_time = start_idx / fps
    clip_duration = num_frames / fps
    
    return frames, start_time, clip_duration

def get_multi_clips(video_path, size=256, clip_len=32, num_clips=3):
    """Extracts multiple uniformly distributed clips from a single video."""
    path_obj = Path(video_path).resolve()
    if not path_obj.exists(): return None

    cap = cv2.VideoCapture(str(path_obj))
    if not cap.isOpened(): return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps): fps = 30.0
    
    if total_frames <= clip_len:
        start_indices = [0]
    else:
        max_start = total_frames - clip_len
        start_indices = np.linspace(0, max_start, num_clips, dtype=int)

    all_clips = []
    for start_idx in start_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        frames = []
        for _ in range(clip_len):
            ret, frame = cap.read()
            if not ret: break
            
            # Use crop/pad to preserve PRNU pixel data (No cv2.resize!)
            h, w = frame.shape[:2]
            start_y = max(0, h // 2 - size // 2)
            start_x = max(0, w // 2 - size // 2)
            frame = frame[start_y : start_y + size, start_x : start_x + size]
            
            if frame.shape[0] < size or frame.shape[1] < size:
                pad_y = max(0, size - frame.shape[0])
                pad_x = max(0, size - frame.shape[1])
                frame = cv2.copyMakeBorder(frame, 0, pad_y, 0, pad_x, cv2.BORDER_CONSTANT, value=[0,0,0])

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        frames = np.array(frames)
        
        # Pad if short
        if len(frames) > 0 and len(frames) < clip_len:
            pad_len = clip_len - len(frames)
            last_frame = frames[-1:]
            padding = np.repeat(last_frame, pad_len, axis=0)
            frames = np.concatenate([frames, padding], axis=0)

        if len(frames) == clip_len:
            start_time = start_idx / fps
            clip_duration = clip_len / fps
            all_clips.append((frames, start_time, clip_duration))
            
    cap.release()
    return all_clips if len(all_clips) > 0 else None

# ==========================================================
# FEATURE EXTRACTION (GPU & CPU)
# ==========================================================

def compute_features_gpu(frames_numpy, device=torch.device('cuda')):
    """CUDA-accelerated feature extraction."""
    t_frames = torch.from_numpy(frames_numpy).float().to(device) / 255.0
    t_frames = t_frames.permute(0, 3, 1, 2) 
    
    T = t_frames.shape[0]
    mid_idx = T // 2

    # Grayscale Conversion
    weights = torch.tensor([0.299, 0.587, 0.114], device=device).view(1, 3, 1, 1)
    gray = (t_frames * weights).sum(dim=1, keepdim=True) 
    
    # Temporal Motion
    diff_seq = torch.abs(gray[1:] - gray[:-1]) 
    diff_mid = torch.abs(gray[mid_idx] - gray[mid_idx + 1] if mid_idx + 1 < T else gray[mid_idx] - gray[mid_idx-1])
    
    # Frequency Analysis (Hardware FFT)
    fft_complex = torch.fft.fft2(gray)
    fft_shift = torch.fft.fftshift(fft_complex, dim =(-2, -1))
    mag_all = torch.log1p(torch.abs(fft_shift))
    mag_avg = mag_all.mean(dim = 0, keepdim = True)
    mag_norm = (mag_avg - mag_avg.min()) / (mag_avg.max() - mag_avg.min() + 1e-6)
    
    # PRNU / Noise Fingerprint (Hardware Convolution)
    k1d = torch.tensor([1, 4, 6, 4, 1], dtype=torch.float32, device=device) / 16.0
    k2d = torch.outer(k1d, k1d).view(1, 1, 5, 5)
    
    g_pad = F.pad(gray, (2, 2, 2, 2), mode='reflect')
    blurred = F.conv2d(g_pad, k2d)

    prnu_all = gray - blurred
    prnu_map = prnu_all.mean(dim=0, keepdim = True) 

    return {
        "rgb_mid": t_frames[mid_idx].unsqueeze(0),
        "rgb_batch": t_frames,
        "diff_seq": diff_seq.unsqueeze(0), 
        "diff": diff_mid.unsqueeze(0).unsqueeze(0),
        "prnu": prnu_map,
        "fft": mag_norm 
    }

def compute_features_cpu(frames, video_path, start_time, clip_duration, device=torch.device("cpu")):
    """Original CPU fallback implementation."""
    frames_norm = frames.astype(np.float32) / 255.0
    mid_idx = len(frames) // 2
    
    gray_stack = np.dot(frames_norm[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)
    diff_stack = np.abs(gray_stack[1:] - gray_stack[:-1])
    
    f = np.fft.fftshift(np.fft.fft2(gray_stack[mid_idx]))
    mag = np.log1p(np.abs(f))
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-6)
    
    g = gray_stack[mid_idx]
    denoised = cv2.GaussianBlur(g, (5, 5), 0)
    prnu_map = g - denoised
    
    spectrogram = extract_audio_spectrogram(video_path, start_time, clip_duration)
    
    return {
        "rgb_batch": torch.from_numpy(frames_norm).permute(0, 3, 1, 2).float().to(device),
        "rgb_mid": torch.from_numpy(frames_norm).permute(0, 3, 1, 2).float()[mid_idx].unsqueeze(0).to(device),
        "diff_seq": torch.from_numpy(diff_stack).unsqueeze(1).unsqueeze(0).float().to(device),
        "prnu": torch.from_numpy(prnu_map).unsqueeze(0).unsqueeze(0).float().to(device),
        "fft": torch.from_numpy(mag).unsqueeze(0).unsqueeze(0).float().to(device),
        "audio": torch.from_numpy(spectrogram).unsqueeze(0).unsqueeze(0).float().to(device)
    }

# ==========================================================
# PROCESSING PIPELINE
# ==========================================================

def process_video_gpu(video_path, output_dir, label, max_frames=32, num_clips=3):
    """
    Main Execution Hook for multiple clips per video via GPU.
    Saves tensors as {video_stem}_clip0.pt, {video_stem}_clip1.pt, etc.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vid_path_obj = Path(video_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    try:
        # Get multiple distributed clips
        clips = get_multi_clips(str(video_path), size=256, clip_len=max_frames, num_clips=num_clips)
        if clips is None: return False
        
        success_count = 0
        
        for i, (frames_np, start_time, clip_duration) in enumerate(clips):
            out_path = Path(output_dir) / f"{vid_path_obj.stem}_clip{i}.pt"
            
            # Process Audio
            audio_np = extract_audio_spectrogram(str(video_path), start_time, clip_duration)
            
            # Process Video Features
            with torch.no_grad():
                features = compute_features_gpu(frames_np, device)
                audio_t = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0).float()

            # Memory Compression
            data_to_save = {
                'rgb_batch': (features['rgb_batch'] * 255).clamp(0, 255).to(torch.uint8).cpu(),
                'prnu':      features['prnu'].squeeze(0).to(torch.float16).cpu(),
                'fft':       features['fft'].squeeze(0).to(torch.float16).cpu(),
                'audio':     audio_t.squeeze(0).to(torch.float16).cpu(),
                'label':     float(label)
            }
            
            torch.save(data_to_save, out_path)
            success_count += 1
            
        return success_count > 0

    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return False