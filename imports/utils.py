import cv2
import numpy as np
import torch
import librosa
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# 1. DATASET PROCESSOR (moved from make_tensor.py)

def process_dataset(input_dir, output_dir, label, max_videos=None):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    videos = list(input_path.rglob("*.mp4")) + list(input_path.rglob("*.avi")) + list(input_path.rglob("*.mov"))
    print(f"Found {len(videos)} videos in {input_dir}")
    
    if max_videos: videos = videos[:max_videos]
    
    success_count = 0
    
    for vid in tqdm(videos):
        try:
            # 1. Extract Frames
            frames = get_frames(str(vid), num_frames=32)
            if frames is None: continue
            
            # 2. Compute Features
            feats = compute_features(frames, str(vid), device=torch.device("cpu"))
            
            # 3. Save to Disk
            save_name = vid.stem + ".pt"
            
            torch.save({
                'rgb_mid':  feats['rgb_mid'].squeeze(0).clone(),   # [3, 256, 256]
                'diff':     feats['diff'].squeeze(0).clone(),      # [1, 256, 256] 
                'diff_seq': feats['diff_seq'].squeeze(0).clone(),  # [31, 1, 256, 256] 
                'prnu':     feats['prnu'].squeeze(0).clone(),      # [1, 256, 256]
                'fft':      feats['fft'].squeeze(0).clone(),       # [1, 256, 256]
                'audio':    feats['audio'].squeeze(0).clone(),     # [1, 128, 128]
                'label':    float(label)
            }, output_path / save_name)
            
            success_count += 1
            
        except Exception as e:
            # print(f"Failed {vid.name}: {e}")
            continue
            
    print(f"Successfully processed {success_count} videos.")


# 2. FRAME EXTRACTION

def get_frames(video_path, size=256, num_frames=32):
    path_obj = Path(video_path).resolve()
    if not path_obj.exists(): return None

    cap = cv2.VideoCapture(str(path_obj))
    if not cap.isOpened(): return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 1: 
        cap.release()
        return None

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if idx in indices:
            frame = cv2.resize(frame, (size, size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            if len(frames) >= num_frames: break
        idx += 1
        
    cap.release()
    frames = np.array(frames)
    
    if len(frames) == 0: return None
    if len(frames) < num_frames:
        pad_len = num_frames - len(frames)
        last_frame = frames[-1:]
        padding = np.repeat(last_frame, pad_len, axis=0)
        frames = np.concatenate([frames, padding], axis=0)
        
    return frames


# 3. FEATURE COMPUTATION

def compute_features(frames, video_path, device=None):
    if device is None: device = torch.device("cpu")

    frames_norm = frames.astype(np.float32) / 255.0
    mid_idx = len(frames) // 2
    
    # Grayscale conversion
    gray_stack = np.dot(frames_norm[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)
    
    # A. Motion Features
    # 1. Sequence: [31, 256, 256]
    diff_stack = np.abs(gray_stack[1:] - gray_stack[:-1]) 
    
    # 2. Single Frame (Original for MoE): [256, 256]
    next_idx = mid_idx + 1 if mid_idx + 1 < len(gray_stack) else mid_idx - 1
    diff_map = np.abs(gray_stack[mid_idx] - gray_stack[next_idx])

    # B. Frequency (FFT)
    f = np.fft.fftshift(np.fft.fft2(gray_stack[mid_idx]))
    mag = np.log1p(np.abs(f))
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-6)
    
    # C. Noise (PRNU)
    prnu_stack = []
    start = max(0, mid_idx-2)
    end = min(len(gray_stack), mid_idx+3)
    for g in gray_stack[start:end]:
        denoised = cv2.GaussianBlur(g, (5, 5), 0)
        prnu_stack.append(g - denoised)
    prnu_map = np.mean(np.array(prnu_stack), axis=0)
    
    # D. Audio
    spectrogram = extract_audio_spectrogram(video_path)
    
    # 1. Main Features
    t_rgb_mid  = torch.from_numpy(frames_norm[mid_idx]).permute(2, 0, 1).unsqueeze(0).float().to(device)
    t_diff_seq = torch.from_numpy(diff_stack).unsqueeze(1).unsqueeze(0).float().to(device) # [1, 31, 1, 256, 256]
    t_diff     = torch.from_numpy(diff_map).unsqueeze(0).unsqueeze(0).float().to(device)    # [1, 1, 256, 256]
    t_prnu     = torch.from_numpy(prnu_map).unsqueeze(0).unsqueeze(0).float().to(device)
    t_fft      = torch.from_numpy(mag).unsqueeze(0).unsqueeze(0).float().to(device)
    t_audio    = torch.from_numpy(spectrogram).unsqueeze(0).unsqueeze(0).float().to(device)

    # [1, T, C, H, W]
    t_rgb_seq = torch.from_numpy(frames_norm).permute(0, 3, 1, 2).unsqueeze(0).float().to(device) 
    # [T, C, H, W]
    t_rgb_batch = torch.from_numpy(frames_norm).permute(0, 3, 1, 2).float().to(device) 

    return {
        "rgb_mid": t_rgb_mid,
        "rgb_seq": t_rgb_seq,       
        "rgb_batch": t_rgb_batch,  
        "diff_seq": t_diff_seq,    
        "diff": t_diff,          
        "prnu": t_prnu,
        "fft": t_fft,
        "audio": t_audio,
        "vis_frames": frames_norm, 
        "vis_audio": spectrogram   
    }

def extract_audio_spectrogram(video_path, target_shape=(128, 128)):
    try:
        y, sr = librosa.load(str(video_path), sr=16000, duration=5.0)
        if len(y) < 1000: return np.zeros(target_shape, dtype=np.float32)

        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=target_shape[0])
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-6)
        mel_spec_resized = cv2.resize(mel_spec_db, (target_shape[1], target_shape[0]))
        
        return mel_spec_resized
    except Exception:
        return np.zeros(target_shape, dtype=np.float32)