"""
Make sure all weights are in the ./models/ directory:
- router_weights.pth
- temporal_lstm.pth
- artifact_model.pth
- noise_model.pth
- frequency_model.pth
- audio_model.pth

If not, run the train script in the same folder

This is a more indepth analysis that takes strides of the whole video for temporal analysis.
Runs slower but accuracy might be increased. A tradeoff for taking the whole video for temporal analysis.
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

# from model_architecture import MoE_Investigator
from model_training.model_architecture import MoE_Investigator
from imports.space import compute_features

def get_multi_clips(video_path, size=256, clip_len=32, num_clips=3):
    path_obj = Path(video_path).resolve()
    if not path_obj.exists(): return None

    cap = cv2.VideoCapture(str(path_obj))
    if not cap.isOpened(): return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= clip_len:
        start_indices = [0]
    else:
        max_start = total_frames - clip_len
        start_indices = np.linspace(0, max_start, num_clips, dtype=int)

    all_clips = []
    
    for start_idx in start_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        frames = []
        for _ in range(clip_len):
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.resize(frame, (size, size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        frames = np.array(frames)
        if len(frames) == clip_len:
            all_clips.append(frames)
            
    cap.release()
    return all_clips if len(all_clips) > 0 else None

class DeepfakeInvestigator:
    def __init__(self, router_path="./models/router_weights.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"initializing on {self.device}")
        
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
            except:
                print("[Router]: Warning - Failed to load router weights.")
        
        self.system.eval()

    def analyze(self, video_path):
        vid_path = Path(video_path)
        print(f"\n{'='*60}")
        print(f"CASE FILE: {vid_path.name}")
        print(f"{'='*60}")
        
        if not vid_path.exists(): print("Error: File not found."); return

        print("\n>> PHASE 1: Rapid Multi-Clip Analysis...")
        verdict_data = self._get_fast_verdict(str(vid_path))
        
        if verdict_data is None:
            print("   Error: Could not process video.")
            return

        print(f"   Verdict: {verdict_data['verdict_text']} ({verdict_data['final_score']:.2%})")
        print("\n>> PHASE 2: Generative Timeline Scan (Searching for Anomalies)...")
        timestamps, suspicions = self._scan_timeline(str(vid_path), stride_sec=0.5)
        
        self._generate_full_report(vid_path.name, verdict_data, timestamps, suspicions)


    def _get_fast_verdict(self, video_path):
        clips = get_multi_clips(video_path, num_clips=3, clip_len=32)
        if clips is None: return None
        
        worst_score = -1.0
        best_data = None
        
        for i, frames in enumerate(clips):
            # Compute Features
            feats = compute_features(frames, video_path,0, 16,device=self.device)
            
            with torch.no_grad():
                # Router
                weights = self.system.router(feats['rgb_mid'])
                w = weights[0].cpu().numpy()
                
                # Experts
                # 1. Temporal
                diff_seq = feats['diff_seq'].squeeze(0) # [31, 1, 256, 256]
                
                # Slide window over this specific clip for the detailed plot
                windows = []
                for j in range(len(diff_seq) - 5 + 1):
                    windows.append(diff_seq[j : j+5])
                
                if len(windows) > 0:
                    batch_win = torch.stack(windows)
                    temp_timeline = torch.sigmoid(self.system.expert_temp(batch_win)).squeeze().cpu().numpy()
                    t_score = float(temp_timeline.max()) if temp_timeline.ndim > 0 else float(temp_timeline)
                else:
                    t_score = 0.5; temp_timeline = np.zeros(30)
                
                # 2. Artifact
                art_logits = self.system.expert_art(feats['rgb_batch'])
                art_masks = torch.sigmoid(art_logits).squeeze(1).cpu().numpy()
                a_score = float(art_masks.max())
                
                # 3. Noise & Freq
                n_score = torch.sigmoid(self.system.expert_noise_head(self.system.expert_noise_net(feats['prnu']))).item()
                f_score = torch.sigmoid(self.system.expert_freq(feats['fft'])).item()
                
                # 4. Audio
                if feats['audio'].max() > 0.01:
                    au_score = torch.sigmoid(self.system.expert_audio(feats['audio'])).item()
                else:
                    au_score = 0.5; w[4] = 0.0

                # Weighted Score
                final = (t_score*w[0] + a_score*w[1] + n_score*w[2] + f_score*w[3] + au_score*w[4]) / (sum(w) + 1e-6)
                
                if final > worst_score:
                    worst_score = final
                    
                    # Prepare Artifact Visualization (Top 5 glitches)
                    frame_scores = np.mean(art_masks, axis=(1, 2))
                    top_indices = np.argsort(frame_scores)[::-1][:5]
                    
                    best_data = {
                        "final_score": final,
                        "verdict_text": "FAKE" if final > 0.5 else "REAL",
                        "scores": [t_score, a_score, n_score, f_score, au_score],
                        "weights": w,
                        "timeline_clip": temp_timeline,
                        "prnu": feats['prnu'].squeeze().cpu().numpy(),
                        "fft": feats['fft'].squeeze().cpu().numpy(),
                        "audio_viz": feats['vis_audio'],
                        "artifacts": (top_indices, art_masks, feats['vis_frames'])
                    }
        
        return best_data

    def _scan_timeline(self, video_path, stride_sec=0.5):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        step_frames = int(fps * stride_sec)
        timestamps = []
        scores = []
        
        for start in range(0, total_frames - 32, step_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            frames = []
            for _ in range(32):
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.resize(frame, (256, 256))
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if len(frames) != 32: continue
            
            frames_arr = np.array(frames)
            feats = compute_features(frames_arr, video_path, start, step_frames, device=self.device)
            
            with torch.no_grad():
                w = self.system.router(feats['rgb_mid'])[0].cpu().numpy()
                diff = feats['diff_seq'].squeeze(0)
                win = diff[10:15].unsqueeze(0)
                t_score = torch.sigmoid(self.system.expert_temp(win)).item()
                a_score = torch.sigmoid(self.system.expert_art(feats['rgb_batch'])).max().item()
                clip_score = (t_score * w[0] + a_score * w[1]) / (w[0] + w[1] + 1e-6)
                
            timestamps.append(start / fps)
            scores.append(clip_score)
            print(f"   Scanning {start/fps:.1f}s | Suspicion: {clip_score:.1%}", end="\r")
            
        cap.release()
        return timestamps, scores

    def _generate_full_report(self, filename, v, time_x, time_y):
        
        score = v['final_score']
        fig = plt.figure(figsize=(20, 16)) # Taller figure
        
        plt.suptitle(f"Deepfake Investigation Unit | Result: {score:.1%} {v['verdict_text']}", 
                     fontsize=20, fontweight='bold', color='red' if score > 0.5 else 'green', y=0.98)
        
        # Grid Layout: 3 Rows
        # Row 1: Detailed Forensics (Temp, Noise, Freq, Audio, Router)
        # Row 2: Artifact Frames (Visual Glitches)
        # Row 3: Full Video Timeline
        gs = gridspec.GridSpec(3, 5, height_ratios=[1, 1.2, 0.8], hspace=0.3)
        
        # 1. Clip Timeline (LSTM)
        ax1 = plt.subplot(gs[0, 0])
        ax1.plot(v['timeline_clip'], color='crimson', linewidth=2, marker='o', markersize=3)
        ax1.set_ylim(0, 1.05)
        ax1.set_title(f"Motion Physics (Worst Clip)\nScore: {v['scores'][0]:.0%}")
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel("Frame Window (0-32)")
        
        # 2. Noise (PRNU)
        ax2 = plt.subplot(gs[0, 1])
        ax2.imshow(v['prnu'], cmap='gray')
        ax2.set_title(f"Sensor Noise\nScore: {v['scores'][2]:.0%}")
        ax2.axis('off')

        # 3. Frequency (FFT)
        ax3 = plt.subplot(gs[0, 2])
        ax3.imshow(v['fft'], cmap='inferno')
        ax3.set_title(f"Frequency Spectrum\nScore: {v['scores'][3]:.0%}")
        ax3.axis('off')

        # 4. Audio
        ax4 = plt.subplot(gs[0, 3])
        ax4.imshow(v['audio_viz'], cmap='inferno', origin='lower', aspect='auto')
        ax4.set_title(f"Audio Analysis\nScore: {v['scores'][4]:.0%}")
        ax4.axis('off')

        # 5. Router Weights
        ax5 = plt.subplot(gs[0, 4])
        labels = ['Motn', 'Art', 'Nois', 'Freq', 'Aud']
        x = np.arange(len(labels))
        ax5.bar(x, v['weights'], color=['blue', 'orange', 'gray', 'purple', 'red'])
        ax5.set_xticks(x); ax5.set_xticklabels(labels)
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

        ax_full = plt.subplot(gs[2, :]) # Spans entire width
        ax_full.plot(time_x, time_y, color='red', linewidth=2)
        
        # Color Zones
        ax_full.fill_between(time_x, time_y, 0, where=[s>=0.5 for s in time_y], color='red', alpha=0.3)
        ax_full.fill_between(time_x, time_y, 0, where=[s<0.5 for s in time_y], color='green', alpha=0.1)
        ax_full.axhline(0.5, color='gray', linestyle='--')
        
        ax_full.set_title("Full Video Scan (Timeline of Suspicion)", fontsize=14, fontweight='bold')
        ax_full.set_xlabel("Video Duration (Seconds)")
        ax_full.set_ylabel("Fake Prob")
        ax_full.set_ylim(0, 1.05)
        ax_full.set_xlim(0, max(time_x) if len(time_x) > 0 else 1)
        ax_full.grid(True, alpha=0.3)
        
        # Add Spikes
        for t, s in zip(time_x, time_y):
            if s > 0.85: ax_full.text(t, s+0.05, "⚠️", ha='center', fontsize=12)

        plt.tight_layout()
        plt.savefig("investigation_report.png")
        plt.show()


if __name__ == "__main__":
    VIDEO_FILE = "data/videos/fake/car.mp4" 
    investigator = DeepfakeInvestigator()
    investigator.analyze(VIDEO_FILE)