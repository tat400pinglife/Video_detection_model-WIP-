import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

from model_architecture import MoE_Investigator
from imports.utils import get_frames, compute_features

class DeepfakeCaseFile:
    def __init__(self, router_path="./models/router_weights.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"initializing on {self.device}")
        
        # Load System
        self.system = MoE_Investigator(
            temp_path="./models/temporal_lstm.pth",  # Updated path
            art_path="./models/artifact_model.pth", 
            noise_path="./models/noise_model.pth",
            freq_path="./models/frequency_model.pth", 
            audio_path="./models/audio_model.pth"
        ).to(self.device)
        
        # Load Router
        if Path(router_path).exists():
            try:
                state = torch.load(router_path, map_location=self.device, weights_only=True)
                self.system.load_state_dict(state, strict=False)
                print(">> [Router]: Intelligence Loaded.")
            except:
                print("[Router]: Warning - Failed to load router weights.")
        
        self.system.eval()

    def analyze(self, video_path):
        vid_path = Path(video_path)
        print(f"\n>> Analyzing Case: {vid_path.name}")
        
        if not vid_path.exists(): print("Error: File not found."); return

        # 1. Extract Frames
        frames = get_frames(str(vid_path))
        if frames is None: return

        # 2. Compute Features
        print("   Extracting forensic traces...")
        feats = compute_features(frames, str(vid_path), device=self.device)
        
        # Unpack Inputs
        rgb_mid   = feats['rgb_mid']
        rgb_batch = feats['rgb_batch'] # For Artifact visualization
        diff_seq  = feats['diff_seq']  # [1, 31, 1, 256, 256] (The full timeline)
        prnu      = feats['prnu']
        fft       = feats['fft']
        audio     = feats['audio']
        
        with torch.no_grad():
            # A. Router Strategy
            # The router looks at the static image (rgb_mid) to decide trust
            weights = self.system.router(rgb_mid)
            w_temp, w_art, w_noise, w_freq, w_audio = weights[0].cpu().numpy()
            
            print(f"   [Router Strategy]:")
            print(f"    - Motion: {w_temp:.1%} | Artifacts: {w_art:.1%} | Noise: {w_noise:.1%}")
            print(f"    - Freq:   {w_freq:.1%} | Audio:     {w_audio:.1%}")

            # B. Run Experts
            
            # 1. TEMPORAL (LSTM Sliding Window)
            # We need to feed the LSTM sequences of 5 frames.
            # We take the full sequence (31 frames) and slide a window over it.
            # Shape: [1, 31, 1, 256, 256] -> Squeeze batch -> [31, 1, 256, 256]
            full_timeline = diff_seq.squeeze(0) 
            windows = []
            SEQ_LEN = 5
            
            # Create windows (e.g. 0-5, 1-6, 2-7...)
            for i in range(len(full_timeline) - SEQ_LEN + 1):
                windows.append(full_timeline[i : i+SEQ_LEN])
            
            if len(windows) > 0:
                # Stack into a batch: [Num_Windows, 5, 1, 256, 256]
                batch_windows = torch.stack(windows)
                
                # Get score for every window
                temp_logits = self.system.expert_temp(batch_windows)
                temp_timeline = torch.sigmoid(temp_logits).squeeze().cpu().numpy()
                
                # The final "Temporal Score" is the MAX anomaly found
                temp_score = float(temp_timeline.max()) if temp_timeline.ndim > 0 else float(temp_timeline)
            else:
                temp_score = 0.5
                temp_timeline = np.zeros(30)

            # 2. Artifacts
            art_logits = self.system.expert_art(rgb_batch)
            art_masks = torch.sigmoid(art_logits).squeeze(1).cpu().numpy()
            art_score = float(art_masks.max()) 
            frame_scores = np.mean(art_masks, axis=(1, 2))
            top_indices = np.argsort(frame_scores)[::-1][:5]

            # 3. Noise
            noise_logits = self.system.expert_noise_head(self.system.expert_noise_net(prnu))
            noise_score = torch.sigmoid(noise_logits).item()

            # 4. Frequency
            freq_logits = self.system.expert_freq(fft)
            freq_score = torch.sigmoid(freq_logits).item()

            # 5. Audio
            if audio.sum() == 0 or audio.max() < 0.01:
                print("   > Audio: Silence detected. Ignoring.")
                audio_score = 0.5; w_audio = 0.0
            else:
                audio_logits = self.system.expert_audio(audio)
                audio_score = torch.sigmoid(audio_logits).item()

            # C. Final Verdict (Weighted Voting)
            final_score = (temp_score * w_temp) + (art_score * w_art) + \
                          (noise_score * w_noise) + (freq_score * w_freq) + \
                          (audio_score * w_audio)
            
            total_w = w_temp + w_art + w_noise + w_freq + w_audio
            if total_w > 0: final_score /= total_w

            verdict = "FAKE" if final_score > 0.5 else "REAL"
            print(f"\n>> FINAL VERDICT: {verdict} ({final_score:.2%})")

            # Visualization Data
            viz_data = {
                "verdict": final_score,
                "scores": [temp_score, art_score, noise_score, freq_score, audio_score],
                "weights": [w_temp, w_art, w_noise, w_freq, w_audio],
                "timeline": temp_timeline, 
                "prnu": prnu.squeeze().cpu().numpy(),
                "fft":  fft.squeeze().cpu().numpy(),
                "audio": feats['vis_audio'],
                "artifacts": (top_indices, art_masks, feats['vis_frames'])
            }
            self.visualize(viz_data)

    def visualize(self, v):
        score = v['verdict']
        fig = plt.figure(figsize=(20, 10))
        plt.suptitle(f"Deepfake Investigation Unit | Result: {score:.1%} FAKE", 
                     fontsize=16, fontweight='bold', color='red' if score > 0.5 else 'green')
        
        gs = gridspec.GridSpec(2, 5, figure=fig, height_ratios=[1, 1.2])

        # TOP ROW
        
        # 1. TEMPORAL LINE PLOT 
        ax1 = plt.subplot(gs[0, 0])
        ax1.plot(v['timeline'], color='crimson', linewidth=2, marker='o', markersize=3)
        ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax1.set_ylim(0, 1.05)
        ax1.set_title(f"Suspicion Timeline (LSTM)\nMax: {v['scores'][0]:.0%}")
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel("Time (Window)")
        
        # 2. Noise
        ax2 = plt.subplot(gs[0, 1])
        ax2.imshow(v['prnu'], cmap='gray')
        ax2.set_title(f"Sensor Noise\nScore: {v['scores'][2]:.0%}")
        ax2.axis('off')

        # 3. Frequency
        ax3 = plt.subplot(gs[0, 2])
        ax3.imshow(v['fft'], cmap='inferno')
        ax3.set_title(f"Frequency (FFT)\nScore: {v['scores'][3]:.0%}")
        ax3.axis('off')

        # 4. Audio
        ax4 = plt.subplot(gs[0, 3])
        ax4.imshow(v['audio'], cmap='inferno', origin='lower', aspect='auto')
        ax4.set_title(f"Audio Spectrum\nScore: {v['scores'][4]:.0%}")
        ax4.axis('off')

        # 5. Router Weights
        ax5 = plt.subplot(gs[0, 4])
        labels = ['Motn', 'Art', 'Nois', 'Freq', 'Aud']
        x = np.arange(len(labels))
        ax5.bar(x, v['weights'], color=['blue', 'orange', 'gray', 'purple', 'red'])
        ax5.set_xticks(x)
        ax5.set_xticklabels(labels)
        ax5.set_title("Router Confidence")
        ax5.set_ylim(0, 1)

        # BOTTOM ROW: ARTIFACT FRAMES
        top_indices, all_masks, all_frames = v['artifacts']
        for i, idx in enumerate(top_indices):
            if i >= 5: break
            ax = plt.subplot(gs[1, i])
            
            mask = all_masks[idx]
            frame = all_frames[idx]
            
            if mask.max() > 0.2:
                # Create Red Heatmap Overlay
                heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                overlay = cv2.addWeighted(np.uint8(255 * frame), 0.7, heatmap, 0.3, 0)
            else:
                overlay = np.uint8(255 * frame)

            ax.imshow(overlay)
            ax.set_title(f"Frame {idx}\nGlitch: {mask.max():.2f}")
            ax.axis('off')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    VIDEO_FILE = "data/videos/fake/car.mp4" 
    investigator = DeepfakeCaseFile()
    investigator.analyze(VIDEO_FILE)