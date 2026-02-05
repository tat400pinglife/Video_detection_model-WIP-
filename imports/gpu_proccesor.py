import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import librosa
from pathlib import Path
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

class VideoTensorizer:
    """
    Modular GPU Tensorizer.
    Designed to integrate with make_tensor.py.
    """

    def __init__(self, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f" >> [System] VideoTensorizer initialized on: {self.device}")
        
        # Pre-calculate Gaussian Kernel for PRNU (Noise Analysis) on GPU
        self.blur_kernel = self._create_gaussian_kernel(kernel_size=5, sigma=1.0).to(self.device)

    def process_video(self, video_path, output_path, label, max_frames=32):
        """
        Processes a single video file and saves the tensor .pt file.
        Matched to the signature called in make_tensor.py.
        
        Args:
            video_path (str): Path to source video.
            output_path (str): Path to save .pt file.
            label (float): 0.0 (Real) or 1.0 (Fake).
            max_frames (int): Number of frames to extract (Default 32).
        
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            # 1. Extract Frames (CPU Step)
            # We enforce the max_frames limit here (e.g., 30 or 32)
            frames_np = self._extract_frames(str(video_path), num_frames=max_frames)
            if frames_np is None: 
                return False

            # 2. Extract Audio (CPU Step)
            audio_np = self._extract_audio(str(video_path))
            
            # 3. Compute Features (GPU Step)
            # Move to GPU: [T, H, W, C] -> [T, C, H, W]
            frames_t = torch.from_numpy(frames_np).float().to(self.device) / 255.0
            frames_t = frames_t.permute(0, 3, 1, 2) 
            
            audio_t = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0).float().to(self.device)

            with torch.no_grad():
                features = self._compute_gpu_features(frames_t)

            # 4. Save to Disk
            # We transfer back to CPU for saving to avoid pickling GPU tensors
            data_to_save = {
                'rgb_mid':  features['rgb_mid'].cpu(),
                'diff_seq': features['diff_seq'].cpu(),
                'diff':     features['diff'].cpu(),
                'prnu':     features['prnu'].cpu(),
                'fft':      features['fft'].cpu(),
                'audio':    audio_t.cpu(),
                'label':    float(label)
            }
            
            torch.save(data_to_save, output_path)
            return True

        except Exception as e:
            print(f"!! Failed to process {Path(video_path).name}: {e}")
            return False

    def _extract_frames(self, video_path, size=256, num_frames=32):
        """
        Extracts exactly 'num_frames' using linspace sampling.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 1: return None

        # Create indices for even sampling
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.resize(frame, (size, size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            
        cap.release()
        
        # Handle padding if video was too short
        frames_arr = np.array(frames)
        if len(frames_arr) < num_frames and len(frames_arr) > 0:
            pad_len = num_frames - len(frames_arr)
            last = frames_arr[-1:]
            padding = np.repeat(last, pad_len, axis=0)
            frames_arr = np.concatenate([frames_arr, padding], axis=0)
            
        return frames_arr if len(frames_arr) == num_frames else None

    def _compute_gpu_features(self, frames):
        """
        Performs forensic math on GPU.
        frames shape: [T, 3, 256, 256]
        """
        # Grayscale weights
        weights = torch.tensor([0.299, 0.587, 0.114], device=self.device).view(1, 3, 1, 1)
        gray = (frames * weights).sum(dim=1, keepdim=True) # [T, 1, 256, 256]
        
        # 1. Temporal Features
        diff_seq = torch.abs(gray[1:] - gray[:-1]) # [T-1, 1, 256, 256]
        
        mid_idx = frames.shape[0] // 2
        diff_mid = torch.abs(gray[mid_idx] - gray[mid_idx + 1] if mid_idx + 1 < len(gray) else gray[mid_idx] - gray[mid_idx-1])
        
        # 2. Frequency (FFT)
        gray_mid = gray[mid_idx, 0]
        fft_complex = torch.fft.fft2(gray_mid)
        fft_shift = torch.fft.fftshift(fft_complex)
        mag = torch.log1p(torch.abs(fft_shift))
        mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-6)
        t_fft = mag.unsqueeze(0).unsqueeze(0)

        # 3. PRNU (Noise)
        # Blur middle 5 frames
        start, end = max(0, mid_idx - 2), min(frames.shape[0], mid_idx + 3)
        gray_slice = gray[start:end]
        
        blurred = F.conv2d(gray_slice, self.blur_kernel, padding=2, groups=1)
        noise_slice = gray_slice - blurred
        prnu_map = noise_slice.mean(dim=0).unsqueeze(0)

        # 4. RGB Mid
        rgb_mid = frames[mid_idx].unsqueeze(0)

        return {
            "rgb_mid": rgb_mid,
            "diff_seq": diff_seq.unsqueeze(0), # [1, T-1, 1, 256, 256]
            "diff": diff_mid.unsqueeze(0).unsqueeze(0),
            "prnu": prnu_map,
            "fft": t_fft
        }

    def _extract_audio(self, video_path):
        try:
            y, sr = librosa.load(video_path, sr=16000, duration=5.0)
            if len(y) < 1000: return np.zeros((128, 128), dtype=np.float32)
            
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
            return cv2.resize(mel_db, (128, 128))
        except:
            return np.zeros((128, 128), dtype=np.float32)

    def _create_gaussian_kernel(self, kernel_size=5, sigma=1.0):
        coords = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
        grid = coords.repeat(kernel_size).view(kernel_size, kernel_size)
        gaussian = torch.exp(-(grid**2 + grid.t()**2) / (2*sigma**2))
        gaussian = gaussian / gaussian.sum()
        return gaussian.view(1, 1, kernel_size, kernel_size)