import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


from model_architecture import MoE_Investigator
from utils import get_frames, compute_features 

class DeepfakeCaseFile:
    def __init__(self, model_path="router_weights.pth"):
        self.device = torch.device("cpu")
        
        # 1. Initialize the Full System
        # We assume the expert weights are already saved in their respective .pth files
        print("Loading Investigator System...")
        self.system = MoE_Investigator(
            temp_path="temporal_model.pth", 
            art_path="unet_artifact_hunter.pth", 
            noise_path="poc_model_256.pth"
        ).to(self.device)
        
        # 2. Load the Trained Router Weights
        # (Only if you have trained the router, otherwise it uses random init)
        try:
            self.system.router.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            print(">> Router Intelligence Loaded.")
        except:
            print(">> Warning: Router weights not found. Using untrained router (heuristic mode).")
        
        self.system.eval()

    def analyze(self, video_path):
        print(f"\n--- Analyzing Case: {Path(video_path).name} ---")
        
        # 1. Process Video
        frames = get_frames(video_path)
        if frames is None: return
        rgb, diff, fft, prnu_mean, prnu_var, all_frames_norm = compute_features(frames)
        
        # 2. Prepare Inputs
        rgb_mid = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()
        rgb_seq = torch.from_numpy(all_frames_norm).permute(0, 3, 1, 2).float().unsqueeze(0)
        t_batch = torch.from_numpy(all_frames_norm).permute(0, 3, 1, 2).float() # For Artifact Batch
        prnu_in = torch.from_numpy(prnu_var).unsqueeze(0).unsqueeze(0).float()

        # 3. Inference
        with torch.no_grad():
            # A. Router
            weights = self.system.router(rgb_mid)
            w_temp, w_art, w_noise = weights[0]
            print(f"Router Strategy: Temp={w_temp:.1%} | Art={w_art:.1%} | Noise={w_noise:.1%}")

            # B. Expert Evidence
            
            # --- Temporal ---
            temp_timeline = torch.sigmoid(self.system.expert_temp(rgb_seq)).squeeze().cpu().numpy()
            temp_score = float(temp_timeline.mean())
            
            # --- Artifact (Top N Logic) ---
            # Run U-Net on ALL frames to find the worst ones
            art_batch_logits = self.system.expert_art(t_batch)
            art_masks = torch.sigmoid(art_batch_logits).squeeze(1).cpu().numpy() # (32, 256, 256)
            
            # Calculate "Suspicion Score" for each frame (sum of active heatmap pixels)
            frame_scores = np.sum(art_masks > 0.5, axis=(1, 2))
            
            # Get Indices of the Top 5 most suspicious frames
            top_n_indices = np.argsort(frame_scores)[::-1][:5]
            art_score = float(art_masks.max())
            
            # --- Noise ---
            noise_score = float(torch.sigmoid(self.system.expert_noise_head(self.system.expert_noise_net(prnu_in))).item())
            
            # C. Verdict
            final_prob = (temp_score * w_temp) + (art_score * w_art) + (noise_score * w_noise)
            final_prob = float(final_prob)

            print(f"Final Verdict: {'FAKE' if final_prob > 0.5 else 'REAL'} ({final_prob:.2%})")
            
            # 4. Visualize with Top 5 Artifacts
            self.visualize(final_prob, temp_timeline, (top_n_indices, art_masks, all_frames_norm), prnu_var)

    def visualize(self, verdict, timeline, artifact_data, prnu_map):
        top_indices, all_masks, all_frames = artifact_data
        
        # Create a grid: Top row for Metrics, Bottom row for Top 5 Frames
        plt.figure(figsize=(18, 9))
        plt.suptitle(f"Deepfake Investigation Results | Verdict: {verdict:.1%} FAKE", fontsize=16, fontweight='bold')

        # --- ROW 1: METRICS ---
        
        # Panel 1: Temporal
        plt.subplot(2, 3, 1)
        plt.plot(timeline, color='crimson', marker='o', linewidth=2)
        plt.axhline(0.5, color='gray', linestyle='--')
        plt.title("Temporal Glitch Timeline")
        plt.ylabel("Fake Probability")
        plt.ylim(0, 1.1)
        plt.grid(alpha=0.3)

        # Panel 2: Noise Pattern
        plt.subplot(2, 3, 2)
        plt.imshow(prnu_map, cmap='gray')
        plt.title("Camera Sensor Noise (PRNU)")
        plt.axis('off')

        # Panel 3: Confidence Bar
        plt.subplot(2, 3, 3)
        plt.bar(["Real", "Fake"], [1-verdict, verdict], color=['green', 'red'], alpha=0.8)
        plt.title(f"Final Weighted Confidence: {verdict:.2%}")
        plt.ylim(0, 1)

        # --- ROW 2: TOP 5 ARTIFACT FRAMES ---
        # We loop through the top 5 suspicious frames found
        for i, idx in enumerate(top_indices):
            # Create Heatmap Overlay
            mask = all_masks[idx]
            frame = all_frames[idx] # (H, W, 3) 0-1 float
            
            # Convert mask to colored heatmap
            heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            # Blend: 70% Original Image + 30% Heatmap
            frame_uint8 = np.uint8(255 * frame)
            overlay = cv2.addWeighted(frame_uint8, 0.7, heatmap, 0.3, 0)

            # mask = all_masks[idx]
            # frame = all_frames[idx]
            
            # # Only create overlay if there is actually something to show
            # if np.sum(mask > 0.5) > 10: 
            #     heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
            #     heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            #     overlay = cv2.addWeighted(np.uint8(255 * frame), 0.7, heatmap, 0.3, 0)
            # else:
            #     # If clean, just show the original frame
            #     overlay = np.uint8(255 * frame)
                
            plt.subplot(2, 5, 6 + i) 
            plt.imshow(overlay)
            # Plot in the second row (using 2x5 grid logic)
            plt.subplot(2, 5, 6 + i) 
            plt.imshow(overlay)
            plt.title(f"Frame {idx}\nSuspicion: {np.sum(mask>0.5):.0f} px")
            plt.axis('off')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Point this to a video file
    case = DeepfakeCaseFile()
    case.analyze("data/videos/maybes/canyon flood.mp4")