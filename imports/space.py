import cv2
import numpy as np
import torch
import librosa
import random
from tqdm import tqdm
from pathlib import Path
import subprocess

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
        # Run ffmpeg and capture the raw audio bytes
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        raw_audio, _ = process.communicate()
        
        # Convert bytes directly to a float32 NumPy array
        audio_array = np.frombuffer(raw_audio, dtype=np.float32)
        return audio_array, sr
    except Exception:
        # Fallback if ffmpeg fails
        return np.zeros(int(sr * duration), dtype=np.float32), sr


def process_dataset(input_dir, output_dir, label, max_videos=None):
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    
    # 1. Create Output Folder if it doesn't exist
    out_path.mkdir(parents=True, exist_ok=True)
    
    # 2. Find Videos
    videos = list(in_path.rglob("*.mp4")) + \
             list(in_path.rglob("*.avi")) + \
             list(in_path.rglob("*.mov"))
    
    if max_videos: videos = videos[:max_videos]
    
    print(f"\nProcessing {len(videos)} videos from: {in_path}")
    print(f"Saving to: {out_path}")
    
    success_count = 0
    
    # 3. Processing Loop
    for vid_path in tqdm(videos, desc="Building Tensors"):
        try:
            save_name = f"{vid_path.stem}.pt"
            save_path = out_path / save_name
            
            # Skip if already exists (Optional: remove this if you want to overwrite)
            #if save_path.exists(): continue

            # A. Extract Frames (Random Clip)
            result = get_frames(str(vid_path), num_frames=32)
            if result is None: continue
            frames, start_time, clip_duration = result
            
            # B. Compute Features (Raw Floats)
            raw_feats = compute_features(frames, str(vid_path), start_time, clip_duration)
            
            # C. Compress Features (The Optimization Step)
            # This converts Float32 -> Uint8/Float16 to save space
            compressed_data = compress_features(raw_feats)
            
            # D. Add Label
            compressed_data['label'] = float(label)
            
            # E. SAVE THE TENSOR (The Critical Step)
            torch.save(compressed_data, save_path)
            
            success_count += 1
            
        except Exception as e:
            # print(f"Failed {vid_path.name}: {e}")
            continue
            
    print(f"Successfully saved {success_count} tensors to {out_path}")
    
def get_frames(video_path, size=256, num_frames=64):
    path_obj = Path(video_path).resolve()
    if not path_obj.exists(): return None

    cap = cv2.VideoCapture(str(path_obj))
    if not cap.isOpened(): return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps): fps = 30.0 # Fallback 
    
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
        
        # This preserves the raw pixel data for PRNU and FFT analysis
        h, w = frame.shape[:2]
        
        # Calculate center crop coordinates
        start_y = max(0, h // 2 - size // 2)
        start_x = max(0, w // 2 - size // 2)
        
        # Slice the image (Crop)
        frame = frame[start_y : start_y + size, start_x : start_x + size]
        
        # Edge case: If the original video is somehow smaller than 256x256, pad it
        if frame.shape[0] < size or frame.shape[1] < size:
            pad_y = max(0, size - frame.shape[0])
            pad_x = max(0, size - frame.shape[1])
            frame = cv2.copyMakeBorder(frame, 0, pad_y, 0, pad_x, cv2.BORDER_CONSTANT, value=[0,0,0])

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        
    cap.release()
    frames = np.array(frames)
    
    # Pad if video is shorter than num_frames
    if len(frames) > 0 and len(frames) < num_frames:
        pad_len = num_frames - len(frames)
        last_frame = frames[-1:]
        padding = np.repeat(last_frame, pad_len, axis=0)
        frames = np.concatenate([frames, padding], axis=0)
        
    if len(frames) == 0: return None
        
    # Calculate exact timestamps
    start_time = start_idx / fps
    clip_duration = num_frames / fps
    
    return frames, start_time, clip_duration

def get_multi_clips(video_path, size=256, clip_len=32, num_clips=3):
    # maybe increase number of clips, a shotgun approach to capture more physics variations in short videos
    path_obj = Path(video_path).resolve()
    if not path_obj.exists(): return None

    cap = cv2.VideoCapture(str(path_obj))
    if not cap.isOpened(): return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
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
            frame = cv2.resize(frame, (size, size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        frames = np.array(frames)
        if len(frames) == clip_len:
            all_clips.append(frames)
            
    cap.release()
    return all_clips if len(all_clips) > 0 else None

def compute_features(frames, video_path, start_time, clip_duration, device=torch.device("cpu")):
    # 1. Prepare Base Images (Float 0.0 - 1.0)
    frames_norm = frames.astype(np.float32) / 255.0
    mid_idx = len(frames) // 2
    
    # 2. Grayscale for Physics Models
    # [32, 256, 256]
    gray_stack = np.dot(frames_norm[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)
    
    # A. Motion (Diff Seq)
    # [31, 256, 256] -> Frame[t+1] - Frame[t]
    diff_stack = np.abs(gray_stack[1:] - gray_stack[:-1])
    
    # B. Frequency (FFT)
    # 2D Fourier Transform on Middle Frame
    f = np.fft.fftshift(np.fft.fft2(gray_stack[mid_idx]))
    mag = np.log1p(np.abs(f))
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-6)
    
    # C. Noise (PRNU)
    # High-Pass Filter on Middle Frame
    g = gray_stack[mid_idx]
    denoised = cv2.GaussianBlur(g, (5, 5), 0)
    prnu_map = g - denoised
    
    # D. Audio
    spectrogram = extract_audio_spectrogram(video_path, start_time, clip_duration)
    
    # We create the full suite needed for Inference
    
    t_rgb_batch = torch.from_numpy(frames_norm).permute(0, 3, 1, 2).float().to(device) # [32, 3, 256, 256]
    t_rgb_mid   = t_rgb_batch[mid_idx].unsqueeze(0)                                    # [1, 3, 256, 256]
    
    t_diff_seq  = torch.from_numpy(diff_stack).unsqueeze(1).unsqueeze(0).float().to(device) # [1, 31, 1, 256, 256]
    
    t_prnu      = torch.from_numpy(prnu_map).unsqueeze(0).unsqueeze(0).float().to(device)   # [1, 1, 256, 256]
    t_fft       = torch.from_numpy(mag).unsqueeze(0).unsqueeze(0).float().to(device)        # [1, 1, 256, 256]
    t_audio     = torch.from_numpy(spectrogram).unsqueeze(0).unsqueeze(0).float().to(device)# [1, 1, 128, 128]

    # For visualization
    vis_frames = frames_norm 
    
    return {
        "rgb_batch": t_rgb_batch,
        "rgb_mid": t_rgb_mid,
        "diff_seq": t_diff_seq,
        "prnu": t_prnu,
        "fft": t_fft,
        "audio": t_audio,
        "vis_frames": vis_frames,
        "vis_audio": spectrogram
    }

def extract_audio_spectrogram(video_path, start_time=0.0, duration=1.06, target_shape=(128, 128)):
    try:
        
        y, sr = load_audio_from_video(str(video_path), start_time, duration, sr=16000)
        
        if len(y) < 1000: return np.zeros(target_shape, dtype=np.float32)
        
        # We STILL use librosa for the actual Mel Spectrogram math
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=target_shape[0])
        # If the audio is completely silent or missing, return zeros
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

def compress_features(feats):
    
    # 1. Compress RGB: Float32 [0-1] -> Uint8 [0-255]
    # This saves 75% space immediately.
    rgb_uint8 = (feats['rgb_batch'] * 255).clamp(0, 255).to(torch.uint8)
    
    # 2. Compress Maps: Float32 -> Float16 (Half Precision)
    # These maps are simple, they don't need 32-bit precision on disk.
    prnu_fp16 = feats['prnu'].squeeze(0).to(torch.float16) # Remove batch dim [1, 1, H, W] -> [1, H, W]
    fft_fp16  = feats['fft'].squeeze(0).to(torch.float16)
    audio_fp16 = feats['audio'].squeeze(0).to(torch.float16)
    
    return {
        'rgb_batch': rgb_uint8.cpu(), # [32, 3, 256, 256]
        'prnu':      prnu_fp16.cpu(), # [1, 256, 256]
        'fft':       fft_fp16.cpu(),  # [1, 256, 256]
        'audio':     audio_fp16.cpu() # [1, 128, 128]
        # Note: 'diff_seq' and 'rgb_mid' are intentionally DROPPED.
        # They will be recalculated in the DataLoader.
    }