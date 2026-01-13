import cv2
import numpy as np
import torch
import librosa
from pathlib import Path

def get_frames(video_path, size=256, num_frames=32):
    """
    Reads a video file and returns a stack of frames.
    Returns: Numpy Array (T, H, W, 3) in uint8 (0-255)
    """
    path_obj = Path(video_path).resolve()
    
    if not path_obj.exists():
        print(f"Error: File not found at {path_obj}")
        return None

    # Force FFmpeg backend
    cap = cv2.VideoCapture(str(path_obj))
    
    if not cap.isOpened():
        print(f"Error: Could not open {path_obj}")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames <= 0: return None
    if fps > 0: total_frames = min(total_frames, int(fps * 8)) # Limit to 8s
    
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    idx, collected = 0, 0
    
    while True:
        ret, frame = cap.read()
        if not ret or idx >= total_frames: break
        
        if idx in indices:
            frame = cv2.resize(frame, (size, size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            collected += 1
        idx += 1
        if collected >= num_frames: break
    cap.release()
    
    frames = np.array(frames)
    if len(frames) < num_frames:
        if len(frames) == 0: return None
        pad = np.zeros((num_frames - len(frames), size, size, 3), dtype=np.uint8)
        frames = np.concatenate([frames, pad], axis=0)
        
    return frames

def extract_audio_spectrogram(video_path, target_shape=(128, 128)):
    """
    Extracts audio from video and converts to Mel-Spectrogram image.
    """
    try:
        # Load audio (first 5 seconds)
        y, sr = librosa.load(str(video_path), sr=16000, duration=5.0)
        
        if len(y) < 1000: # Silent or too short
            return np.zeros(target_shape, dtype=np.float32)

        # Create Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=target_shape[0])
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to 0-1
        mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-6)
        
        # Resize to fixed shape
        mel_spec_resized = cv2.resize(mel_spec_db, (target_shape[1], target_shape[0]))
        
        return mel_spec_resized
    except Exception:
        # Fail gracefully if no audio track
        return np.zeros(target_shape, dtype=np.float32)

def compute_features(frames, video_path, device=None):
    """
    Calculates visual and audio features and converts them to Tensors.
    """
    # Default to CPU if not specified
    if device is None:
        device = torch.device("cpu")

    # --- 1. Visual Pre-processing ---
    frames_norm = frames.astype(np.float32) / 255.0
    mid_idx = len(frames) // 2
    gray_stack = np.dot(frames_norm[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)
    
    # FFT
    f = np.fft.fftshift(np.fft.fft2(gray_stack[mid_idx]))
    mag = np.log1p(np.abs(f))
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-6)
    
    # PRNU
    prnu_stack = []
    for g in gray_stack:
        denoised = cv2.GaussianBlur(g, (5, 5), 0)
        prnu_stack.append(g - denoised)
    prnu_var = np.var(np.array(prnu_stack), axis=0)

    # --- 2. Audio Pre-processing ---
    spectrogram = extract_audio_spectrogram(video_path)
    
    # --- 3. Tensor Conversion ---
    
    t_rgb_mid = torch.from_numpy(frames_norm[mid_idx]).permute(2, 0, 1).unsqueeze(0).float().to(device)
    t_rgb_seq = torch.from_numpy(frames_norm).permute(0, 3, 1, 2).unsqueeze(0).float().to(device)
    t_rgb_batch = torch.from_numpy(frames_norm).permute(0, 3, 1, 2).float().to(device)
    t_prnu = torch.from_numpy(prnu_var).unsqueeze(0).unsqueeze(0).float().to(device)
    t_audio = torch.from_numpy(spectrogram).unsqueeze(0).unsqueeze(0).float().to(device)

    return {
        "rgb_mid": t_rgb_mid,
        "rgb_seq": t_rgb_seq,
        "rgb_batch": t_rgb_batch,
        "prnu": t_prnu,
        "audio": t_audio,
        "vis_frames": frames_norm, # Keep raw for plotting
        "vis_audio": spectrogram   # Keep raw for plotting
    }