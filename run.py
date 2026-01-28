import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Imports
from model_architecture import MoE_Investigator
from utils import get_frames, compute_features

class DeepfakeCaseFile:
    def __init__(self, model_path="./models/router_weights.pth"):
        # 1. Force CPU for local testing
        self.device = torch.device("cpu")
        print(f">> Device Set: {self.device} (Local Mode)")
        
        # 2. Initialize System
        # Note: If you haven't trained Audio/Router yet, it will use random weights for them.
        print("Loading Investigator System...")
        self.system = MoE_Investigator(
            temp_path="./models/temporal_model.pth", 
            art_path="./models/unet_artifact_hunter.pth", 
            noise_path="./models/poc_model_256.pth",
            audio_path="./models/audio_expert.pth" #
        ).to(self.device)
        
        # 3. Load Router Weights
        try:
            self.system.router.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            print(">> Router Intelligence Loaded.")
        except:
            print(">> Warning: Router weights not found or mismatch. Using untrained router (random strategy).")
        
        self.system.eval()

    def analyze(self, video_path):
        print(f"\n--- Analyzing Case: {Path(video_path).name} ---")
        
        # 1. Get Data (Returns Tensors on CPU)
        frames = get_frames(video_path)
        if frames is None: return
               
        # Pass video_path so it can extract audio
        data = compute_features(frames, video_path, device=self.device)
        
        # Extract Inputs
        rgb_mid = data["rgb_mid"]
        rgb_seq = data["rgb_seq"]
        rgb_batch = data["rgb_batch"]
        prnu_in = data["prnu"]
        audio_in = data["audio"]
        
        with torch.no_grad():
            # A. Router (Outputs 4 weights)
            weights = self.system.router(rgb_mid)
            w_temp, w_art, w_noise, w_audio = weights[0]
            
            print(f"Router Strategy:")
            print(f"  [Temp]: {w_temp:.1%} | [Art]: {w_art:.1%} | [Noise]: {w_noise:.1%} | [Audio]: {w_audio:.1%}")

            # B. Experts Execution
            
            # --- Temporal Expert ---
            temp_logits = self.system.expert_temp(rgb_seq)
            temp_timeline = torch.sigmoid(temp_logits).squeeze().cpu().numpy() 
            temp_score = float(temp_timeline.mean())

            # --- Artifact Expert ---
            art_batch_logits = self.system.expert_art(rgb_batch)
            art_masks = torch.sigmoid(art_batch_logits).squeeze(1).cpu().numpy()
            
            # Find Top 5 Suspicious Frames
            frame_scores = np.sum(art_masks > 0.5, axis=(1, 2))
            top_indices = np.argsort(frame_scores)[::-1][:5]
            art_score = float(art_masks.max())
            
            # --- Noise Expert ---
            noise_logits = self.system.expert_noise_head(self.system.expert_noise_net(prnu_in))
            noise_score = float(torch.sigmoid(noise_logits).item())

            # --- Audio Expert ---
            if audio_in.sum() == 0:
                print("  > No audio track detected (Skipping Audio Expert).")
                audio_score = 0.5
            else:
                audio_logits = self.system.expert_audio(audio_in)
                audio_score = float(torch.sigmoid(audio_logits).item())
                print(f"  > Audio Suspicion: {audio_score:.2%}")

            # C. Verdict
            final_prob = (temp_score*w_temp) + (art_score*w_art) + (noise_score*w_noise) + (audio_score*w_audio)
            
            print(f"Final Verdict: {'FAKE' if final_prob > 0.5 else 'REAL'} ({final_prob:.2%})")
            
            # D. Prepare Vis Data
            vis_frames = data["vis_frames"]
            vis_prnu   = prnu_in.squeeze().cpu().numpy()
            vis_audio  = data["vis_audio"]

            # Visualize
            self.visualize(final_prob, temp_timeline, (top_indices, art_masks, vis_frames), vis_prnu, vis_audio)

    def visualize(self, verdict, timeline, artifact_data, prnu_map, audio_spec):
        import matplotlib.gridspec as gridspec # Import this locally or at top
        
        top_indices, all_masks, all_frames = artifact_data
        
        # Setup Figure
        fig = plt.figure(figsize=(18, 9))
        plt.suptitle(f"Deepfake Investigation Results | Verdict: {verdict:.1%} FAKE", fontsize=16, fontweight='bold')
        
        # Use GridSpec to handle different column counts
        # We use 20 columns (LCM of 4 and 5) to align everything perfectly
        gs = gridspec.GridSpec(2, 20, figure=fig)

        # --- ROW 1: METRICS (4 Plots) ---
        # Each plot takes 5 units of width (20 / 4 = 5)
        
        # 1. Temporal
        ax1 = plt.subplot(gs[0, 0:5])
        ax1.plot(timeline, color='crimson', marker='o')
        ax1.axhline(0.5, color='gray', linestyle='--')
        ax1.set_title("Temporal Glitch Timeline")
        ax1.set_ylim(0, 1.1)
        ax1.grid(alpha=0.3)

        # 2. Noise
        ax2 = plt.subplot(gs[0, 5:10])
        ax2.imshow(prnu_map, cmap='gray')
        ax2.set_title("Camera Noise (PRNU)")
        ax2.axis('off')
        
        # 3. Audio
        ax3 = plt.subplot(gs[0, 10:15])
        ax3.imshow(audio_spec, cmap='inferno', origin='lower')
        ax3.set_title("Audio Spectrum")
        ax3.axis('off')

        # 4. Confidence
        ax4 = plt.subplot(gs[0, 15:20])
        ax4.bar(["Real", "Fake"], [1-verdict, verdict], color=['green', 'red'], alpha=0.8)
        ax4.set_title(f"Confidence: {verdict:.2%}")
        ax4.set_ylim(0, 1)

        # --- ROW 2: TOP 5 ARTIFACT FRAMES ---
        # Each plot takes 4 units of width (20 / 5 = 4)
        for i, idx in enumerate(top_indices):
            col_start = i * 4
            ax = plt.subplot(gs[1, col_start : col_start + 4])
            
            mask = all_masks[idx]
            frame = all_frames[idx]
            
            # Overlay Logic
            if np.sum(mask > 0.5) > 10:
                heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                overlay = cv2.addWeighted(np.uint8(255 * frame), 0.7, heatmap, 0.3, 0)
            else:
                overlay = np.uint8(255 * frame)

            ax.imshow(overlay)
            ax.set_title(f"Frame {idx}\nScore: {np.sum(mask>0.5):.0f}")
            ax.axis('off')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Point this to a video file
    case = DeepfakeCaseFile()
    # Ensure this path is correct
    case.analyze("data/videos/maybes/cat and rat.mp4")