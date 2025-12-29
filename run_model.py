import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path

# note to self: probably should have a file just for architecture that can be ported in instead of copy pasting from model code

# frame extraction
def get_frames(video_path, size=256, num_frames=32):
    video_path = Path(video_path).resolve()
    cap = cv2.VideoCapture(str(video_path))
    frames = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if total_frames <= 0:
        cap.release()
        return None

    if fps > 0:
        total_frames = min(total_frames, int(fps * 8))

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    idx = 0
    collected = 0

    while True:
        ret, frame = cap.read()
        if not ret or idx >= total_frames:
            break

        if idx in indices:
            frame = cv2.resize(frame, (size, size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            collected += 1

        idx += 1
        if collected >= num_frames:
            break

    cap.release()

    frames = np.array(frames)
    if len(frames) < num_frames:
        pad = np.zeros((num_frames - len(frames), size, size, 3), dtype=np.uint8)
        frames = np.concatenate([frames, pad], axis=0)

    return frames
def extract_prnu_residual(gray):
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    residual = gray - denoised
    return residual

# model architecture
class SimpleBranch(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            # --- Layer 1 (general neighborhood noise) ---
            # Input: 256x256
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # Reduces size by 2 -> 128x128
            
            # --- Layer 2 (frame texture) ---
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # Reduces size by 2 -> 64x64

            # --- Layer 3 (regional features/consistency) ---
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # Reduces size by 2 -> 32x32

            #  --- Flatten ---
            nn.Flatten() 
        )
        
    def forward(self, x):
        return self.net(x)
    
class PRNUBranch(nn.Module):
    #prnu as a classifier becomes meaningless when dealing with larger quantities of signals
    # this is lowered with a gate such that it is only used when necessary
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten()
        )

    def forward(self, x):
        return self.net(x)

class TinyDeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.rgb_branch = SimpleBranch(in_channels=3)
        self.diff_branch = SimpleBranch(in_channels=1)
        self.fft_branch = SimpleBranch(in_channels=1)
        self.prnu_branch = PRNUBranch()
        
        # MATH: 
        # Final image size is 32x32. 
        # Final depth is 64 channels.
        # 64 * 32 * 32 = 65,536 features per branch.
        self.feature_size = 64 * 32 * 32 
        self.prnu_size = 32 * 32 * 32 
        
        self.prnu_gate = nn.Sequential(
            nn.Linear(self.prnu_size, 1),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            # 65,536 * 4 branches = 262,144 inputs
            nn.Linear(self.feature_size * 3 + self.prnu_size * 2, 128), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1) 
        )

    def forward(self, rgb, diff, fft, prnu_mean, prnu_var):
        f_rgb = self.rgb_branch(rgb)
        f_diff = self.diff_branch(diff)
        f_fft = self.fft_branch(fft)

        f_prnu_mean = self.prnu_branch(prnu_mean)
        f_prnu_var = self.prnu_branch(prnu_var)

        gate = self.prnu_gate(f_prnu_mean)
        f_prnu_mean = f_prnu_mean * gate

        combined = torch.cat([f_rgb, f_diff, f_fft, f_prnu_mean, f_prnu_var], dim=1)
        return self.classifier(combined)

# frame processing
def compute_features(frames):
    # Normalize RGB to 0-1
    frames_norm = frames.astype(np.float32) / 255.0
    
    # Select Middle Frame for RGB/FFT/Diff Analysis
    mid_idx = len(frames) // 2
    rgb = frames_norm[mid_idx] # (256, 256, 3)

    # Grayscale conversion for entire stack (needed for temporal PRNU)
    gray_stack = np.dot(frames_norm[..., :3], [0.299, 0.587, 0.114]) # (T, 256, 256)
    
    # 1. Temporal Diff (Middle Frame)
    diff = np.abs(gray_stack[mid_idx] - gray_stack[mid_idx - 1])
    
    # 2. FFT (Middle Frame)
    f = np.fft.fftshift(np.fft.fft2(gray_stack[mid_idx]))
    mag = np.log1p(np.abs(f))
    # Normalize FFT 0-1
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-6)

    # 3. PRNU (Temporal Calculation)
    # We must compute PRNU for ALL frames to get the Variance Map
    prnu_stack = []
    for i in range(len(gray_stack)):
        r = extract_prnu_residual(gray_stack[i])
        prnu_stack.append(r)
    
    prnu_stack = np.array(prnu_stack, dtype=np.float32) # (32, 256, 256)

    # Aggregate across time (Axis 0)
    prnu_mean = np.mean(prnu_stack, axis=0) # (256, 256)
    prnu_var = np.var(prnu_stack, axis=0)   # (256, 256) <- SPATIAL MAP, NOT SCALAR

    return rgb, diff, mag, prnu_mean, prnu_var

# model inference
def predict_video(model_path, video_path):
    device = torch.device("cpu")

    model = TinyDeepfakeDetector().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    frames = get_frames(video_path)
    if frames is None:
        print("Could not load video.")
        return

    rgb, diff, fft, prnu_mean, prnu_var = compute_features(frames)

    t_rgb = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    t_diff = torch.from_numpy(diff).unsqueeze(0).unsqueeze(0)
    t_fft = torch.from_numpy(fft).unsqueeze(0).unsqueeze(0)
    t_prnu_mean = torch.from_numpy(prnu_mean).unsqueeze(0).unsqueeze(0)
    t_prnu_var = torch.from_numpy(prnu_var).unsqueeze(0).unsqueeze(0)
   
    with torch.no_grad():
        out = model(t_rgb.float(), t_diff.float(), t_fft.float(), t_prnu_mean.float(), t_prnu_var.float())
        score = torch.sigmoid(out).item()

    label = "FAKE (AI)" if score > 0.5 else "REAL"
    conf = score if score > 0.5 else 1 - score

    print(f"Result: {label}")
    print(f"Confidence: {conf:.2%}")
    print(f"Raw score: {score:.4f}")


if __name__ == "__main__":
    MODEL_FILE = "poc_model_256.pth"
    TEST_VIDEO = "data/videos/maybes/crackhead.mp4"
    predict_video(MODEL_FILE, TEST_VIDEO)
