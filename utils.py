import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class DeepfakeGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, inputs):
        self.model.zero_grad()
        output = self.model(*inputs)
        target = output[0]
        target.backward()
        
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.detach()
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = np.maximum(heatmap.cpu().numpy(), 0)
        if np.max(heatmap) > 0: heatmap /= np.max(heatmap)
        return heatmap

def visualize_gradcam(rgb_frame, heatmap, title="Grad-CAM Attention"):
    heatmap = cv2.resize(heatmap, (rgb_frame.shape[1], rgb_frame.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    frame_uint8 = np.uint8(255 * rgb_frame)
    overlay = cv2.addWeighted(frame_uint8, 0.6, heatmap_color, 0.4, 0)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.title(title)
    plt.axis("off")
    plt.show()

def get_frames(video_path, size=256, num_frames=32):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames <= 0:
        cap.release()
        return np.zeros((num_frames, size, size, 3), dtype=np.uint8)

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
    """
    Extracts RGB, Diff, FFT, and PRNU features.
    Input: Numpy array (T, H, W, 3) in uint8 (0-255).
    """
    # 1. Standardize to float32 (0.0 - 1.0)
    frames_norm = frames.astype(np.float32) / 255.0
    mid_idx = len(frames) // 2
    rgb = frames_norm[mid_idx]
    
    # 2. Grayscale (Float32)
    gray_stack = np.dot(frames_norm[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)
    
    # 3. Diff (Motion)
    diff = np.abs(gray_stack[mid_idx] - gray_stack[mid_idx - 1])
    
    # 4. FFT (Frequency)
    f = np.fft.fftshift(np.fft.fft2(gray_stack[mid_idx]))
    mag = np.log1p(np.abs(f))
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-6)
    
    # 5. PRNU (Sensor Noise)
    prnu_stack = []
    for g in gray_stack:
        denoised = cv2.GaussianBlur(g, (5, 5), 0)
        prnu_stack.append(g - denoised)
    prnu_stack = np.array(prnu_stack, dtype=np.float32)
    
    prnu_mean = np.mean(prnu_stack, axis=0)
    prnu_var = np.var(prnu_stack, axis=0)
    
    return rgb, diff, mag, prnu_mean, prnu_var, frames_norm