import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
assert torch.cuda.is_available()

from model_architecture import MoE_Investigator
from utils_gpu import get_frames, compute_features 

class DeepfakeCaseFile:
    def __init__(self, model_path="router_weights.pth"):

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f">> GPU Detected: {torch.cuda.get_device_name(0)}")
            # Enable Benchmark Mode (optimizes C++ kernels for your specific GPU)
            torch.backends.cudnn.benchmark = True 
        else:
            self.device = torch.device("cpu")
            print(">> Warning: No GPU found.")

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
        
        # 1. Get Data (Returns Tensors on GPU)
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
            # We need the timeline for visualization
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
                print("  > No audio track detected.")
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
        top_indices, all_masks, all_frames = artifact_data
        
        # Grid: 2 Rows. Top for metrics, Bottom for artifacts.
        plt.figure(figsize=(18, 9))
        plt.suptitle(f"Deepfake Investigation Results | Verdict: {verdict:.1%} FAKE", fontsize=16, fontweight='bold')

        # --- ROW 1: METRICS ---
        
        # 1. Temporal
        plt.subplot(2, 4, 1)
        plt.plot(timeline, color='crimson', marker='o')
        plt.axhline(0.5, color='gray', linestyle='--')
        plt.title("Temporal Glitch Timeline")
        plt.ylim(0, 1.1)

        # 2. Noise
        plt.subplot(2, 4, 2)
        plt.imshow(prnu_map, cmap='gray')
        plt.title("Camera Noise (PRNU)")
        plt.axis('off')
        
        # 3. Audio (NEW)
        plt.subplot(2, 4, 3)
        plt.imshow(audio_spec, cmap='inferno', origin='lower')
        plt.title("Audio Spectrum")
        plt.axis('off')

        # 4. Confidence
        plt.subplot(2, 4, 4)
        plt.bar(["Real", "Fake"], [1-verdict, verdict], color=['green', 'red'], alpha=0.8)
        plt.title(f"Confidence: {verdict:.2%}")
        plt.ylim(0, 1)

        # --- ROW 2: TOP 5 ARTIFACT FRAMES ---
        for i, idx in enumerate(top_indices):
            mask = all_masks[idx]
            frame = all_frames[idx]
            
            # Only overlay if suspicion exists
            if np.sum(mask > 0.5) > 10:
                heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                overlay = cv2.addWeighted(np.uint8(255 * frame), 0.7, heatmap, 0.3, 0)
            else:
                overlay = np.uint8(255 * frame)

            # Plot in positions 5, 6, 7, 8, 9
            plt.subplot(2, 5, 6 + i) 
            plt.imshow(overlay)
            plt.title(f"Frame {idx}\nScore: {np.sum(mask>0.5):.0f}")
            plt.axis('off')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Point this to a video file
    case = DeepfakeCaseFile()
    case.analyze("data/videos/maybes/canyon flood.mp4")