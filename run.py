"""
Unified Deepfake Investigation Script
Scans the entire video sequentially, builds the anomaly timeline, 
and extracts the most suspicious clip for detailed forensic reporting.
"""

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

from model_training.model_architecture import MoE_Investigator
from imports.combined import compute_features_gpu, extract_audio_spectrogram


class DeepfakeInvestigator:
    def __init__(self, router_path="./models/router_weights.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing Investigation Unit on {self.device}...")
        
        # Load System
        self.system = MoE_Investigator(
            temp_path="./models/temporal_lstm.pth", 
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
            except Exception as e:
                print(f"[Router]: Warning - Failed to load router weights: {e}")
        else:
            print("[Router]: No router weights found! Using untrained initialization.")
            
        self.system.eval()

    def analyze(self, video_path, stride_sec=1):
        vid_path = Path(video_path)
        print(f"\n{'='*60}")
        print(f"CASE FILE: {vid_path.name}")
        print(f"{'='*60}")
        
        if not vid_path.exists(): 
            print("Error: File not found.")
            return

        print(f"\n>> Initiating Comprehensive Scan (Stride: {stride_sec}s)...")
        timestamps, suspicions, worst_case_data = self._comprehensive_scan(str(vid_path), stride_sec)
        
        if not timestamps:
            print("   Error: Could not process video.")
            return

        # Print the final verdict based on the worst anomaly found
        final_score = worst_case_data['final_score']
        print(f"\n>> SCAN COMPLETE.")
        print(f"   Peak Suspicion Found: {final_score:.2%} ({worst_case_data['verdict_text']})")
        print("   Generating Forensic Report...")
        
        self._generate_full_report(vid_path.name, worst_case_data, timestamps, suspicions)


    def _comprehensive_scan(self, video_path, stride_sec=1):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps == 0 or np.isnan(fps): 
            fps = 30.0
            
        step_frames = int(fps * stride_sec)
        
        timestamps = []
        scores = []
        
        worst_score = -1.0
        best_data = None
        
        # Slide through the entire video
        for start in range(0, total_frames - 32, step_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            frames = []
            
            # Extract 32 frames for this specific window
            for _ in range(32):
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.resize(frame, (256, 256))
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if len(frames) != 32: 
                continue
            
            frames_arr = np.array(frames)
            
            # 1. Timeline Audio Extraction
            start_sec = start / fps
            clip_dur = 32 / fps
            audio_np = extract_audio_spectrogram(video_path, start_sec, clip_dur)
            audio_t = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0).float().to(self.device)
            
            # 2. Timeline GPU Feature Extraction
            feats = compute_features_gpu(frames_arr, device=self.device)
            
            with torch.no_grad():
                # 3. ROUTER CALCULATION
                logits = self.system.router(feats['rgb_mid'])
                w = torch.softmax(logits, dim=1)[0].cpu().numpy()
                
                # 4. EXPERT SCORES
                # Temporal
                diff = feats['diff_seq'].squeeze(0)
                windows = [diff[j : j+5] for j in range(len(diff) - 4)]
                
                if len(windows) > 0:
                    batch_win = torch.stack(windows)
                    temp_timeline = torch.sigmoid(self.system.expert_temp(batch_win)).squeeze().cpu().numpy()
                    t_score = float(temp_timeline.max()) if temp_timeline.ndim > 0 else float(temp_timeline)
                else:
                    t_score = 0.5; temp_timeline = np.zeros(30)
                
                # Artifact
                art_masks_tensor = torch.sigmoid(self.system.expert_art(feats['rgb_batch']))
                a_score = art_masks_tensor.view(art_masks_tensor.size(0), -1).topk(500, dim=1)[0].mean(dim=1).max().item()
                art_masks = art_masks_tensor.squeeze(1).cpu().numpy()
                
                # PRNU & Frequency (Corrected: No extra unsqueeze)
                n_score = torch.sigmoid(self.system.expert_noise_head(self.system.expert_noise_net(feats['prnu']))).item()
                f_score = torch.sigmoid(self.system.expert_freq(feats['fft'])).item()
                
                # Audio Integration
                if audio_np.max() > 0.01:
                    au_score = torch.sigmoid(self.system.expert_audio(audio_t)).item()
                else:
                    au_score = 0.5
                    w[3] = 0.0  # Mute Audio weight
                    w = w / (w.sum() + 1e-6) # Re-normalize remaining weights

                # 5. FINAL COMBINATION
                clip_score = (t_score*w[0] + a_score*w[1] + n_score*w[2] + au_score*w[3] + f_score*w[4])
                
            # Log the timeline data
            timestamps.append(start_sec)
            scores.append(clip_score)
            print(f"   Scanning {start_sec:05.1f}s | Suspicion: {clip_score:05.1%} | Weights: [M:{w[0]:.1f}, A:{w[1]:.1f}, N:{w[2]:.1f}, Au:{w[3]:.1f}, F:{w[4]:.1f}]", end="\r")
            
            # 6. CAPTURE THE WORST OFFENDER
            if clip_score > worst_score:
                worst_score = clip_score
                
                # Calculate the 5 worst visual frames for the Artifact visualizer
                frame_scores = np.mean(art_masks, axis=(1, 2))
                top_indices = np.argsort(frame_scores)[::-1][:5]
                
                best_data = {
                    "final_score": clip_score,
                    "verdict_text": "FAKE" if clip_score > 0.5 else "REAL",
                    "scores": [t_score, a_score, n_score, f_score, au_score],
                    "weights": w,
                    "timeline_clip": temp_timeline,
                    "prnu": feats['prnu'].squeeze().cpu().numpy(),
                    "fft": feats['fft'].squeeze().cpu().numpy(),
                    "audio_viz": audio_np,
                    "artifacts": (top_indices, art_masks, frames_arr)
                }
                
        cap.release()
        # Clean up the console line
        print(" "*100, end="\r")
        return timestamps, scores, best_data

    def _generate_full_report(self, filename, v, time_x, time_y):
        score = v['final_score']
        fig = plt.figure(figsize=(20, 16)) 
        
        plt.suptitle(f"Deepfake Investigation Unit | Result: {score:.1%} {v['verdict_text']}", 
                     fontsize=20, fontweight='bold', color='red' if score > 0.5 else 'green', y=0.98)
        
        gs = gridspec.GridSpec(3, 5, height_ratios=[1, 1.2, 0.8], hspace=0.3)
        
        ax1 = plt.subplot(gs[0, 0])
        ax1.plot(v['timeline_clip'], color='crimson', linewidth=2, marker='o', markersize=3)
        ax1.set_ylim(0, 1.05)
        ax1.set_title(f"Motion Physics (Worst Clip)\nScore: {v['scores'][0]:.0%}")
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel("Frame Window (0-32)")
        
        ax2 = plt.subplot(gs[0, 1])
        ax2.imshow(v['prnu'], cmap='gray')
        ax2.set_title(f"Sensor Noise\nScore: {v['scores'][2]:.0%}")
        ax2.axis('off')

        ax3 = plt.subplot(gs[0, 2])
        ax3.imshow(v['fft'], cmap='inferno')
        ax3.set_title(f"Frequency Spectrum\nScore: {v['scores'][3]:.0%}")
        ax3.axis('off')

        ax4 = plt.subplot(gs[0, 3])
        ax4.imshow(v['audio_viz'], cmap='inferno', origin='lower', aspect='auto')
        ax4.set_title(f"Audio Analysis\nScore: {v['scores'][4]:.0%}")
        ax4.axis('off')

        ax5 = plt.subplot(gs[0, 4])
        labels = ['Motn', 'Art', 'Nois', 'Aud', 'Freq']
        x = np.arange(len(labels))
        ax5.bar(x, v['weights'], color=['blue', 'orange', 'gray', 'red', 'purple'])
        ax5.set_xticks(x); ax5.set_xticklabels(labels)
        ax5.set_title("Router Confidence")
        ax5.set_ylim(0, 1)

        top_indices, all_masks, all_frames = v['artifacts']
        for i, idx in enumerate(top_indices):
            if i >= 5: break
            ax = plt.subplot(gs[1, i])
            
            mask = all_masks[idx]
            frame = all_frames[idx]
            
            if mask.max() > 0.2:
                heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                overlay = cv2.addWeighted(np.uint8(255 * frame), 0.7, heatmap, 0.3, 0)
            else:
                overlay = np.uint8(255 * frame)

            ax.imshow(overlay)
            ax.set_title(f"Frame {idx}\nGlitch: {mask.max():.2f}")
            ax.axis('off')

        ax_full = plt.subplot(gs[2, :])
        ax_full.plot(time_x, time_y, color='red', linewidth=2)
        
        ax_full.fill_between(time_x, time_y, 0, where=[s>=0.5 for s in time_y], color='red', alpha=0.3)
        ax_full.fill_between(time_x, time_y, 0, where=[s<0.5 for s in time_y], color='green', alpha=0.1)
        ax_full.axhline(0.5, color='gray', linestyle='--')
        
        ax_full.set_title(f"Full Video Scan (Timeline of Suspicion)", fontsize=14, fontweight='bold')
        ax_full.set_xlabel("Video Duration (Seconds)")
        ax_full.set_ylabel("Fake Prob")
        ax_full.set_ylim(0, 1.05)
        ax_full.set_xlim(0, max(time_x) if len(time_x) > 0 else 1)
        ax_full.grid(True, alpha=0.3)
        
        for t, s in zip(time_x, time_y):
            if s > 0.85: ax_full.text(t, s+0.05, "⚠️", ha='center', fontsize=12)

        plt.tight_layout()
        plt.savefig(f"./result_pngs/investigation_report_{filename}.png")
        print(">> Report saved as 'investigation_report.png'")
        plt.show()


if __name__ == "__main__":
    VIDEO_FILE = "data/videos/fake/Turkey ring.mp4" 
    investigator = DeepfakeInvestigator()
    investigator.analyze(VIDEO_FILE)