import torch
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

# may not use this

def compute_fft(frames):
    gray = np.dot(frames[..., :3], [0.299, 0.587, 0.114])
    ffts = []
    for i in range(len(gray)):
        f = np.fft.fft2(gray[i])
        fshift = np.fft.fftshift(f)
        magnitude = np.log1p(np.abs(fshift))
        m_min, m_max = magnitude.min(), magnitude.max()
        if m_max - m_min > 1e-5:
            magnitude = (magnitude - m_min) / (m_max - m_min)
        else:
            magnitude = np.zeros_like(magnitude)
        ffts.append(magnitude)
    return np.array(ffts, dtype=np.float32)

def compute_diff(frames):
    gray = np.dot(frames[..., :3], [0.299, 0.587, 0.114])
    diffs = [np.zeros_like(gray[0])]
    for i in range(1, len(gray)):
        d = np.abs(gray[i] - gray[i-1])
        diffs.append(d)
    return np.array(diffs, dtype=np.float32) / 255.0

def compute_prnu_residuals(frames):
    # Convert to grayscale float32
    gray = np.dot(frames[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)
    residuals = []
    for i in range(len(gray)):
        img = gray[i]
        denoised = cv2.GaussianBlur(img, (5, 5), 0)
        noise = img - denoised
        residuals.append(noise)
    return np.array(residuals, dtype=np.float32) / 255.0

# AUGMENTATION LOGIC
SOURCE_DIR = Path("./data/frames")

def save_augmented_sample(frames_np, original_label, save_path):
    """
    Takes a stack of augmented RGB frames (0-255 uint8 or 0-1 float),
    recalculates ALL features, and saves a new .pt file.
    """
    # 1. Ensure input is proper 0-255 uint8 for consistent feature calculation
    if frames_np.dtype != np.uint8:
        # Assume it's float 0-1, convert to uint8
        frames_uint8 = (frames_np * 255).astype(np.uint8)
    else:
        frames_uint8 = frames_np

    # 2. Recalculate Features
    fft = compute_fft(frames_uint8)
    diff = compute_diff(frames_uint8)
    prnu = compute_prnu_residuals(frames_uint8)
    
    # 3. Normalize RGB for storage (0-1 float32)
    rgb_norm = frames_uint8.astype(np.float32) / 255.0

    # 4. Pack Data (Match the original structure)
    new_data = {
        "rgb": torch.from_numpy(rgb_norm).permute(0, 3, 1, 2),  # (T, 3, H, W)
        "fft": torch.from_numpy(fft).unsqueeze(1),              # (T, 1, H, W)
        "diff": torch.from_numpy(diff).unsqueeze(1),            # (T, 1, H, W)
        "prnu": torch.from_numpy(prnu).unsqueeze(1),            # (T, 1, H, W)
        "label": original_label                                 # Keep original label
    }
    
    torch.save(new_data, save_path)

def augment_data():
    files = list(SOURCE_DIR.rglob("*.pt"))
    print(f"Augmenting {len(files)} files with feature recalculation...")

    for path in tqdm(files):
        if "_aug_" in path.name: continue # Skip files we already augmented
        
        try:
            data = torch.load(path, weights_only=False)
            
            # Extract Original RGB
            # Shape is (T, 3, H, W) -> Convert to (T, H, W, 3) for OpenCV
            t_frames = data['rgb']
            frames = t_frames.permute(0, 2, 3, 1).numpy() # Float32 0-1
            
            base_name = path.stem
            parent = path.parent
            label = data['label']

            # Horizontal Flip
            # Good for making the model ignore position bias
            frames_flip = np.array([cv2.flip(f, 1) for f in frames])
            save_augmented_sample(frames_flip, label, parent / f"{base_name}_aug_flip.pt")
            
            # Gaussian Noise
            # Forces model to ignore random camera grain and look for structural noise
            noise = np.random.normal(0, 0.05, frames.shape).astype(np.float32)
            frames_noise = np.clip(frames + noise, 0, 1)
            save_augmented_sample(frames_noise, label, parent / f"{base_name}_aug_noise.pt")
            
            # Darker Lighting 
            # Forces model to ignore "brightness" as a feature
            frames_dark = np.clip(frames * 0.6, 0, 1)
            save_augmented_sample(frames_dark, label, parent / f"{base_name}_aug_dark.pt")

        except Exception as e:
            print(f"Skipped {path.name}: {e}")

if __name__ == "__main__":
    augment_data()