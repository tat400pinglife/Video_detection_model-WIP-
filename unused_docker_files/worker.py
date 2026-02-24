import os
import matplotlib
# CRITICAL FIX: Set backend to 'Agg' BEFORE importing pyplot.
# This prevents it from trying to open a window on the server (Docker).
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import cv2
import numpy as np
import shutil
from pathlib import Path
from celery import Celery
import warnings

# --- CUSTOM IMPORTS ---
from model_architecture import MoE_Investigator
from imports.utils import get_frames, compute_features

# Suppress warnings
warnings.filterwarnings("ignore")

redis_url = 'redis://redis:6379/0'
celery_app = Celery('deepfake_worker', broker=redis_url, backend=redis_url)

# Docker Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "./models" 

print(f"[INFO] Initializing MoE System on {DEVICE}...")

# Global system variable
system = None

def load_models():
    global system
    try:
        system = MoE_Investigator(
            temp_path=f"{MODEL_DIR}/temporal_lstm.pth",
            art_path=f"{MODEL_DIR}/artifact_model.pth", 
            noise_path=f"{MODEL_DIR}/noise_model.pth",
            freq_path=f"{MODEL_DIR}/frequency_model.pth", 
            audio_path=f"{MODEL_DIR}/audio_model.pth"
        ).to(DEVICE)

        router_path = f"{MODEL_DIR}/router_weights.pth"
        if os.path.exists(router_path):
            # Load router weights (map_location is vital for CPU/Docker)
            system.router.load_state_dict(
                torch.load(router_path, map_location=DEVICE, weights_only=True)
            )
            print(">> Router Intelligence Loaded.")
        else:
            print(">> Warning: Router weights missing. Using random strategy.")
        
        system.eval()
        print("[INFO] System Ready.")
        return True

    except Exception as e:
        print(f"[CRITICAL] Failed to load models: {e}")
        return False

# Attempt to load immediately on startup
load_models()

@celery_app.task(bind=True)
def analyze_task(self, video_path):
    print(f"\n--- Analyzing Case: {Path(video_path).name} ---")
    
    # Reload models if they failed or were lost
    if system is None:
        if not load_models():
            return {"status": "Failed", "error": "AI Models failed to load."}

    try:
        # 1. Get Data
        frames = get_frames(video_path)
        if frames is None: 
            return {"status": "Failed", "error": "Could not extract frames"}
            
        # 2. Extract Features
        print("   Extracting forensic traces...")
        # device=DEVICE ensures tensors are created on correct device
        feats = compute_features(frames, video_path, device=DEVICE)
        
        # Unpack Inputs
        rgb_mid   = feats["rgb_mid"]
        rgb_batch = feats["rgb_batch"]
        diff_seq  = feats["diff_seq"] # [1, 31, 1, 256, 256] -> The sequence for LSTM
        prnu      = feats["prnu"]
        fft       = feats["fft"]
        audio     = feats["audio"]
        
        with torch.no_grad():
            # A. Router Strategy
            weights = system.router(rgb_mid)
            w_temp, w_art, w_noise, w_freq, w_audio = weights[0].cpu().numpy()
            
            # B. Experts Execution
            
            # 1. TEMPORAL (LSTM Sliding Window)
            # We need to slice the 31-frame sequence into windows of 5 frames
            full_timeline_tensor = diff_seq.squeeze(0) # [31, 1, 256, 256]
            SEQ_LEN = 5
            windows = []
            
            # Sliding Window Logic
            if len(full_timeline_tensor) >= SEQ_LEN:
                for i in range(len(full_timeline_tensor) - SEQ_LEN + 1):
                    windows.append(full_timeline_tensor[i : i+SEQ_LEN])
                
                # Stack into batch: [Num_Windows, 5, 1, 256, 256]
                batch_windows = torch.stack(windows).to(DEVICE)
                
                # Run LSTM
                temp_logits = system.expert_temp(batch_windows)
                temp_probs = torch.sigmoid(temp_logits).squeeze().cpu().numpy()
                
                # Handle edge case (single window result)
                if temp_probs.ndim == 0: temp_probs = np.array([temp_probs])
                
                temp_score = float(temp_probs.max())
                temp_timeline = temp_probs # For visualization
            else:
                # Video too short
                temp_score = 0.5
                temp_timeline = np.zeros(10)

            # 2. Artifacts
            art_logits = system.expert_art(rgb_batch)
            art_masks = torch.sigmoid(art_logits).squeeze(1).cpu().numpy()
            art_score = float(art_masks.max())
            
            # Top 5 Frames for Viz
            frame_scores = np.mean(art_masks, axis=(1, 2))
            top_indices = np.argsort(frame_scores)[::-1][:5]
            
            # 3. Noise
            # Note: In MoE class, noise is split. We call head(net(x))
            noise_feat = system.expert_noise_net(prnu)
            noise_logits = system.expert_noise_head(noise_feat)
            noise_score = float(torch.sigmoid(noise_logits).item())

            # 4. Frequency (FFT)
            freq_logits = system.expert_freq(fft)
            freq_score = float(torch.sigmoid(freq_logits).item())

            # 5. Audio
            if audio.sum() == 0 or audio.max() < 0.01:
                print("   > Audio: Silence detected/ignored.")
                audio_score = 0.5
                w_audio = 0.0 # Force weight to zero
            else:
                audio_logits = system.expert_audio(audio)
                audio_score = float(torch.sigmoid(audio_logits).item())

            # C. Verdict
            final_prob = (temp_score * w_temp) + \
                         (art_score * w_art) + \
                         (noise_score * w_noise) + \
                         (freq_score * w_freq) + \
                         (audio_score * w_audio)
            
            total_w = w_temp + w_art + w_noise + w_freq + w_audio
            if total_w > 0: final_prob /= total_w
            
            verdict_text = 'FAKE' if final_prob > 0.5 else 'REAL'

            # D. Visualization
            vis_data = {
                "verdict": final_prob,
                "scores": [temp_score, art_score, noise_score, freq_score, audio_score],
                "weights": [w_temp, w_art, w_noise, w_freq, w_audio],
                "timeline": temp_timeline,
                "prnu": prnu.squeeze().cpu().numpy(),
                "fft":  fft.squeeze().cpu().numpy(),
                "audio": feats["vis_audio"],
                "artifacts": (top_indices, art_masks, feats["vis_frames"])
            }

            # Save report alongside video
            report_filename = f"{Path(video_path).stem}_report.png"
            report_dir = os.path.dirname(video_path)
            report_path = os.path.join(report_dir, report_filename)
            
            save_visual_report(report_path, vis_data)

        print(f"[INFO] Analysis Complete. Verdict: {verdict_text} ({final_prob:.2%})")

        return {
            "file": video_path,
            "prediction": verdict_text,
            "confidence": round(float(final_prob) * 100, 2),
            "breakdown": {
                "temporal": round(temp_score, 2),
                "artifact": round(art_score, 2),
                "noise": round(noise_score, 2),
                "frequency": round(freq_score, 2),
                "audio": round(audio_score, 2)
            },
            "report_image": report_filename
        }

    except Exception as e:
        print(f"[ERROR] Processing Error: {e}")
        return {"status": "Failed", "error": str(e)}

def save_visual_report(save_path, v):
    print(f"[DEBUG] Generating report image at: {save_path}")
    
    try:
        score = v['verdict']
        fig = plt.figure(figsize=(20, 10))
        plt.suptitle(f"Forensic Analysis | Verdict: {score:.1%} FAKE", 
                     fontsize=16, fontweight='bold', color='red' if score > 0.5 else 'green')
        
        gs = gridspec.GridSpec(2, 5, figure=fig, height_ratios=[1, 1.2])

        # --- ROW 1: METRICS ---
        
        # 1. Timeline
        ax1 = plt.subplot(gs[0, 0])
        ax1.plot(v['timeline'], color='crimson', marker='o', markersize=3)
        ax1.axhline(0.5, color='gray', linestyle='--')
        ax1.set_ylim(0, 1.1)
        ax1.set_title(f"Suspicion Timeline\nMax: {v['scores'][0]:.0%}")
        ax1.grid(alpha=0.3)

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

        # 5. Weights
        ax5 = plt.subplot(gs[0, 4])
        labels = ['Motn', 'Art', 'Nois', 'Freq', 'Aud']
        x = np.arange(len(labels))
        ax5.bar(x, v['weights'], color=['blue', 'orange', 'gray', 'purple', 'red'])
        ax5.set_xticks(x)
        ax5.set_xticklabels(labels)
        ax5.set_title("Router Strategy")
        ax5.set_ylim(0, 1)

        # --- ROW 2: ARTIFACTS ---
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

        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        if os.path.exists(save_path):
            print(f"[SUCCESS] Report saved.")
        else:
            print(f"[ERROR] File not found after saving.")

    except Exception as e:
        print(f"[CRITICAL ERROR] Matplotlib failed: {e}")