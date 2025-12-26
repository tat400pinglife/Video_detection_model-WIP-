import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path

# --- 1. MODEL ARCHITECTURE (3-Layer Version) ---
class SimpleBranch(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten() 
        )
    def forward(self, x): return self.net(x)

class TinyDeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_branch = SimpleBranch(3)
        self.diff_branch = SimpleBranch(1)
        self.fft_branch = SimpleBranch(1)
        self.feature_size = 64 * 32 * 32 
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size * 3, 128), 
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 1) 
        )
    def forward(self, rgb, diff, fft):
        combined = torch.cat([self.rgb_branch(rgb), self.diff_branch(diff), self.fft_branch(fft)], dim=1)
        return self.classifier(combined)

# --- 2. VIDEO PROCESSING UTILITIES ---
def get_frames(video_path, size=256, num_frames=32):
    # This fixes the "isOpened? False" error on Windows
    video_path = Path(video_path).resolve()
    
    # --- Start of your code ---
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames <= 0:
        cap.release()
        return None # Changed from returning zeros to None for inference safety

    if fps > 0:
        total_frames = min(total_frames, int(fps * 8))

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    current_idx = 0
    collected = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if current_idx >= total_frames: break

        if current_idx in indices:
            frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            collected += 1
        current_idx += 1
        if collected >= num_frames: break
            
    cap.release()
    
    frames = np.array(frames)
    if len(frames) < num_frames:
        padding = np.zeros((num_frames - len(frames), size, size, 3), dtype=np.uint8)
        if len(frames) > 0: frames = np.concatenate([frames, padding], axis=0)
        else: frames = padding

            
    return frames

def compute_features(frames):
    idx = 16 
    rgb = frames[idx].astype(np.float32) / 255.0
    gray_stack = np.dot(frames[..., :3], [0.299, 0.587, 0.114])
    
    f = np.fft.fft2(gray_stack[idx])
    fshift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))
    m_min, m_max = magnitude.min(), magnitude.max()
    fft = (magnitude - m_min) / (m_max - m_min) if (m_max - m_min) > 1e-5 else np.zeros_like(magnitude)
    
    diff = np.abs(gray_stack[idx] - gray_stack[idx-1]) / 255.0
    
    return rgb, diff, fft

# --- 3. INFERENCE LOOP ---
def predict_video(model_path, video_path):
    device = torch.device("cpu")
    model = TinyDeepfakeDetector().to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except Exception as e:
        print(f"Model Load Error: {e}")
        return

    model.eval()

    print(f"Processing: {video_path}...")
    frames = get_frames(video_path)
    
    if frames is None:
        print("Error: Could not read video (frames returned None).")
        return

    rgb, diff, fft = compute_features(frames)
    
    t_rgb = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()
    t_diff = torch.from_numpy(diff).unsqueeze(0).unsqueeze(0).float()
    t_fft = torch.from_numpy(fft).unsqueeze(0).unsqueeze(0).float()
    
    with torch.no_grad():
        output = model(t_rgb, t_diff, t_fft)
        score = torch.sigmoid(output).item()
        
    label = "FAKE (AI)" if score > 0.5 else "REAL"
    confidence = score if score > 0.5 else 1 - score
    
    print(f"Result: {label}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Raw Score: {score:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    MODEL_FILE = "poc_model_256.pth"
    # Make sure this path exists relative to where you run the script
    TEST_VIDEO = "data/videos/maybes/crackhead.mp4" 
    predict_video(MODEL_FILE, TEST_VIDEO)
