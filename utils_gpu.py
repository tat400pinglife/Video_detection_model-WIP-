import cv2
import numpy as np
import torch
from pathlib import Path
import librosa # <--- Make sure to import this

def extract_audio_spectrogram(video_path, target_shape=(128, 128)):
    try:
        # Load audio (only first 5 seconds to match video logic)
        y, sr = librosa.load(str(video_path), sr=16000, duration=5.0)
        
        if len(y) < 1000: # Silent or too short
            return np.zeros(target_shape, dtype=np.float32)

        # Create Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=target_shape[0])
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to 0-1
        mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-6)
        
        # Resize to fixed shape (128x128)
        # Note: In reality, stretching time axis is bad practice for music, 
        # but for short deepfake clips, it works as a texture map.
        mel_spec_resized = cv2.resize(mel_spec_db, (target_shape[1], target_shape[0]))
        
        return mel_spec_resized
    except Exception as e:
        # If no audio track found
        return np.zeros(target_shape, dtype=np.float32)

def compute_features(frames, video_path, device=None): # <--- NOTE: Added video_path arg
    # ... [Keep existing image code] ...
    
    # 1. Existing Image Processing
    frames_norm = frames.astype(np.float32) / 255.0
    mid_idx = len(frames) // 2
    gray_stack = np.dot(frames_norm[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)
    
    # FFT
    f = np.fft.fftshift(np.fft.fft2(gray_stack[mid_idx]))
    mag = np.log1p(np.abs(f)); mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-6)
    
    # PRNU
    prnu_stack = []
    for g in gray_stack: prnu_stack.append(g - cv2.GaussianBlur(g, (5, 5), 0))
    prnu_var = np.var(np.array(prnu_stack), axis=0)

    # 2. NEW: Audio Processing
    spectrogram = extract_audio_spectrogram(video_path) # (128, 128) numpy
    
    # 3. Tensors
    if device is None: device = torch.device("cpu")
    
    t_rgb_mid = torch.from_numpy(frames_norm[mid_idx]).permute(2, 0, 1).unsqueeze(0).float().to(device)
    t_rgb_seq = torch.from_numpy(frames_norm).permute(0, 3, 1, 2).unsqueeze(0).float().to(device)
    t_rgb_batch = torch.from_numpy(frames_norm).permute(0, 3, 1, 2).float().to(device)
    t_prnu = torch.from_numpy(prnu_var).unsqueeze(0).unsqueeze(0).float().to(device)
    
    # Audio Tensor: (1, 1, 128, 128)
    t_audio = torch.from_numpy(spectrogram).unsqueeze(0).unsqueeze(0).float().to(device)

    return {
        "rgb_mid": t_rgb_mid,
        "rgb_seq": t_rgb_seq,
        "rgb_batch": t_rgb_batch,
        "prnu": t_prnu,
        "audio": t_audio, # <--- New Key
        "vis_frames": frames_norm,
        "vis_audio": spectrogram # <--- For plotting
    }