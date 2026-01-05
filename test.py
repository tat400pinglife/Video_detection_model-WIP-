import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from model_architecture import TinyDeepfakeDetector, ArtifactSegmentor, TemporalDetector
from gradcam_utils import DeepfakeGradCAM, visualize_gradcam

# flags
RUN_CLASSIFIER = True      # Step 1: Real vs Fake Check
RUN_GRADCAM    = True      # Step 2: Show where the Classifier looked
RUN_SEGMENTATION = True    # Step 3: Use U-Net to find specific artifacts
RUN_TEMPORAL   = True      # Step 4: Check for frame-to-frame glitches

# paths
VIDEO_PATH     = "data/videos/maybes/cat and rat.mp4"
MODEL_CLF      = "poc_model_256.pth"
MODEL_SEG      = "unet_artifact_hunter.pth"
MODEL_TEMP     = "temporal_model.pth"

# data processing
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

def compute_features(frames):
    frames_norm = frames.astype(np.float32) / 255.0
    mid_idx = len(frames) // 2
    rgb = frames_norm[mid_idx]
    
    gray_stack = np.dot(frames_norm[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)
    diff = np.abs(gray_stack[mid_idx] - gray_stack[mid_idx - 1])
    
    f = np.fft.fftshift(np.fft.fft2(gray_stack[mid_idx]))
    mag = np.log1p(np.abs(f))
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-6)
    
    prnu_stack = []
    for g in gray_stack:
        denoised = cv2.GaussianBlur(g, (5, 5), 0)
        prnu_stack.append(g - denoised)
    prnu_stack = np.array(prnu_stack, dtype=np.float32)
    
    prnu_mean = np.mean(prnu_stack, axis=0)
    prnu_var = np.var(prnu_stack, axis=0)
    
    return rgb, diff, mag, prnu_mean, prnu_var, frames_norm

def visualize_top_artifacts(frames, masks, indices, scores):
    num_show = len(indices)
    cols = 5
    rows = (num_show + cols - 1) // cols
    plt.figure(figsize=(15, 3 * rows))
    plt.suptitle("Top Suspicious Frames (Artifact Segmentation)", fontsize=14)
    for i, idx in enumerate(indices):
        plt.subplot(rows, cols, i + 1)
        heatmap = cv2.applyColorMap(np.uint8(255 * masks[idx]), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(np.uint8(255 * frames[idx]), 0.7, heatmap, 0.3, 0)
        plt.imshow(overlay)
        plt.title(f"Frame {idx}\nScore: {scores[idx]:.1f}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Main Analysis Pipeline
def main():
    device = torch.device("cpu")
    print(f"--- Analyzing: {Path(VIDEO_PATH).name} ---")
    
    frames = get_frames(VIDEO_PATH)
    if frames is None:
        print("Error: Could not load video.")
        return
        
    rgb, diff, fft, prnu_mean, prnu_var, all_frames_norm = compute_features(frames)
    
    # Prepare Tensors (Float32), error with 64
    t_rgb = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()
    t_diff = torch.from_numpy(diff).unsqueeze(0).unsqueeze(0).float()
    t_fft = torch.from_numpy(fft).unsqueeze(0).unsqueeze(0).float()
    t_p_mean = torch.from_numpy(prnu_mean).unsqueeze(0).unsqueeze(0).float()
    t_p_var = torch.from_numpy(prnu_var).unsqueeze(0).unsqueeze(0).float()
    

    is_fake = False
    if RUN_CLASSIFIER:
        try:
            clf = TinyDeepfakeDetector().to(device)
            clf.load_state_dict(torch.load(MODEL_CLF, map_location=device, weights_only=True))
            clf.eval()
            
            with torch.no_grad():
                score = torch.sigmoid(clf(t_rgb, t_diff, t_fft, t_p_mean, t_p_var)).item()
            
            print(f"\n[1] CLASSIFIER RESULT")
            print(f"    Confidence: {score:.2%} ({'FAKE' if score > 0.5 else 'REAL'})")
            is_fake = score > 0.5
        except Exception as e:
            print(f"    Skipping Classifier: {e}")


    if RUN_GRADCAM and RUN_CLASSIFIER: # Needs classifier to be loaded
        print(f"\n[2] RUNNING GRAD-CAM")
        try:
            # We need the model instance again with gradients enabled
            target_layer = clf.rgb_branch.net[6] # Conv 32->64
            cam = DeepfakeGradCAM(clf, target_layer)
            heatmap = cam.generate_heatmap((t_rgb, t_diff, t_fft, t_p_mean, t_p_var))
            
            # Display
            rgb_display = t_rgb.squeeze().permute(1, 2, 0).numpy()
            visualize_gradcam(rgb_display, heatmap, title="Grad-CAM (Global Attention)")
        except Exception as e:
            print(f"    Grad-CAM Error: {e}")


    if RUN_SEGMENTATION:
        print(f"\n[3] RUNNING ARTIFACT SEGMENTATION")
        if is_fake or not RUN_CLASSIFIER: # Run if fake OR if we skipped classifier
            try:
                seg = ArtifactSegmentor().to(device)
                seg.load_state_dict(torch.load(MODEL_SEG, map_location=device, weights_only=True))
                seg.eval()
                
                batch = torch.from_numpy(all_frames_norm).permute(0, 3, 1, 2).float().to(device)
                with torch.no_grad():
                    masks = torch.sigmoid(seg(batch)).squeeze(1).cpu().numpy()
                
                # Rank frames
                frame_scores = np.sum(masks > 0.5, axis=(1, 2))
                top_indices = np.argsort(frame_scores)[::-1][:10]
                suspicious = [idx for idx in top_indices if frame_scores[idx] > 5]
                
                if suspicious:
                    visualize_top_artifacts(all_frames_norm, masks, suspicious, frame_scores)
                else:
                    print("    No distinct artifacts found.")
            except Exception as e:
                print(f"    Skipping Segmentation: {e}")
        else:
            print("    Skipping: Video classified as REAL.")

    if RUN_TEMPORAL:
        print(f"\n[4] RUNNING TEMPORAL ANALYSIS")
        try:
            temp_model = TemporalDetector().to(device)
            temp_model.load_state_dict(torch.load(MODEL_TEMP, map_location=device, weights_only=True))
            temp_model.eval()
            
            # Prepare Sequence: (1, 32, 3, 256, 256)
            seq_batch = torch.from_numpy(all_frames_norm).permute(0, 3, 1, 2).float().unsqueeze(0).to(device)
            
            with torch.no_grad():
                frame_probs = torch.sigmoid(temp_model(seq_batch)).squeeze().cpu().numpy()
            
            # Plot Timeline
            plt.figure(figsize=(10, 3))
            plt.plot(frame_probs, marker='o', color='r')
            plt.axhline(0.5, color='gray', linestyle='--')
            plt.title("Temporal Glitch Probability (Frame-by-Frame)")
            plt.xlabel("Frame Index")
            plt.ylabel("Fake Score")
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
             print(f"    Skipping Temporal: {e}")

if __name__ == "__main__":
    main()