import os
import matplotlib
# CRITICAL FIX: Set backend to 'Agg' BEFORE importing pyplot.
# This prevents it from trying to open a window on the server.
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

DEVICE = torch.device("cpu") # Change to "cuda" if available in Docker
MODEL_DIR = "./models" 

print(f"[INFO] Initializing MoE System on {DEVICE}...")

system = None

try:
    # Initialize the Full 5-Expert System
    system = MoE_Investigator(
        temp_path=f"{MODEL_DIR}/temporal_model.pth", 
        art_path=f"{MODEL_DIR}/unet_artifact_hunter.pth", 
        noise_path=f"{MODEL_DIR}/poc_model_256.pth",
        freq_path=f"{MODEL_DIR}/frequency_model.pth",  # <--- NEW
        audio_path=f"{MODEL_DIR}/audio_expert.pth"
    ).to(DEVICE)

    router_path = f"{MODEL_DIR}/router_weights.pth"
    if os.path.exists(router_path):
        system.router.load_state_dict(torch.load(router_path, map_location=DEVICE, weights_only=True))
        print(">> Router Intelligence Loaded.")
    else:
        print(">> Warning: Router weights missing. Using random strategy.")
    
    system.eval()
    print("[INFO] System Ready.")

except Exception as e:
    print(f"[CRITICAL] Failed to load models: {e}")
    system = None

@celery_app.task(bind=True)
def analyze_task(self, video_path):
    print(f"\n--- Analyzing Case: {Path(video_path).name} ---")
    
    if system is None:
        return {"status": "Failed", "error": "AI Models failed to load."}

    try:
        # 1. Get Data
        frames = get_frames(video_path)
        if frames is None: 
            return {"status": "Failed", "error": "Could not extract frames"}
            
        # 2. Extract Features (RGB, Diff, FFT, PRNU, Audio)
        print("   Extracting forensic traces...")
        feats = compute_features(frames, video_path, device=DEVICE)
        
        # Unpack Inputs
        rgb_mid = feats["rgb_mid"]
        rgb_batch = feats["rgb_batch"]
        diff    = feats["diff"]
        prnu    = feats["prnu"]
        fft     = feats["fft"]
        audio   = feats["audio"]
        
        with torch.no_grad():
            # A. Router Strategy
            weights = system.router(rgb_mid)
            # Unpack 5 weights
            w_temp, w_art, w_noise, w_freq, w_audio = weights[0].cpu().numpy()
            
            # B. Experts Execution
            
            # 1. Temporal (Motion)
            # Run on sequence of diffs for timeline
            gray_frames = np.dot(feats['vis_frames'][..., :3], [0.299, 0.587, 0.114])
            diff_stack = []
            for i in range(len(gray_frames) - 1):
                d = np.abs(gray_frames[i] - gray_frames[i+1])
                diff_stack.append(d)
            diff_stack = np.array(diff_stack)
            
            t_diff_seq = torch.from_numpy(diff_stack).unsqueeze(1).float().to(DEVICE)
            
            temp_logits = system.expert_temp(t_diff_seq)
            temp_timeline = torch.sigmoid(temp_logits).squeeze().cpu().numpy()
            temp_score = float(temp_timeline.max())

            # 2. Artifacts
            art_logits = system.expert_art(rgb_batch)
            art_masks = torch.sigmoid(art_logits).squeeze(1).cpu().numpy()
            art_score = float(art_masks.max())
            
            # Top 5 Frames
            frame_scores = np.mean(art_masks, axis=(1, 2))
            top_indices = np.argsort(frame_scores)[::-1][:5]
            
            # 3. Noise
            noise_logits = system.expert_noise_head(system.expert_noise_net(prnu))
            noise_score = float(torch.sigmoid(noise_logits).item())

            # 4. Frequency (FFT)
            freq_logits = system.expert_freq(fft)
            freq_score = float(torch.sigmoid(freq_logits).item())

            # 5. Audio
            if audio.sum() == 0 or audio.max() < 0.01:
                print(">> No audio track detected (or silence).")
                audio_score = 0.5
                w_audio = 0.0 # Force weight to 0
            else:
                audio_logits = system.expert_audio(audio)
                audio_score = float(torch.sigmoid(audio_logits).item())

            # C. Verdict
            final_prob = (temp_score * w_temp) + \
                         (art_score * w_art) + \
                         (noise_score * w_noise) + \
                         (freq_score * w_freq) + \
                         (audio_score * w_audio)
            
            # Normalize if weights changed
            total_w = w_temp + w_art + w_noise + w_freq + w_audio
            if total_w > 0: final_prob /= total_w
            
            verdict_text = 'FAKE' if final_prob > 0.5 else 'REAL'

            # D. Visualization (Save to Disk)
            vis_data = {
                "verdict": final_prob,
                "scores": [temp_score, art_score, noise_score, freq_score, audio_score],
                "weights": [w_temp, w_art, w_noise, w_freq, w_audio],
                "timeline": temp_timeline,
                "diff": diff.squeeze().cpu().numpy(),
                "prnu": prnu.squeeze().cpu().numpy(),
                "fft":  fft.squeeze().cpu().numpy(),
                "audio": feats["vis_audio"],
                "artifacts": (top_indices, art_masks, feats["vis_frames"])
            }

            # Generate Report Filename
            report_filename = f"{Path(video_path).stem}_report.png"
            # Save to /app/uploads folder
            report_path = os.path.join(os.path.dirname(video_path), report_filename)
            
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