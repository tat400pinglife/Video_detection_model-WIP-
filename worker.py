import os
import matplotlib
# CRITICAL FIX: Set backend to 'Agg' BEFORE importing pyplot.
# This prevents it from trying to open a window on the server.
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import torch
import cv2
import numpy as np
import shutil
from pathlib import Path
from celery import Celery

# --- CUSTOM IMPORTS ---
from model_architecture import MoE_Investigator
from utils import get_frames, compute_features

redis_url = 'redis://redis:6379/0'
celery_app = Celery('deepfake_worker', broker=redis_url, backend=redis_url)

DEVICE = torch.device("cpu") # Change to "cuda" if using NVIDIA Docker
MODEL_DIR = "./models" 

print(f"[INFO] Initializing MoE System on {DEVICE}...")

try:
    system = MoE_Investigator(
        temp_path=f"{MODEL_DIR}/temporal_model.pth", 
        art_path=f"{MODEL_DIR}/unet_artifact_hunter.pth", 
        noise_path=f"{MODEL_DIR}/poc_model_256.pth",
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
            
        # 2. Extract Features
        data = compute_features(frames, video_path, device=DEVICE)
        
        # Unpack Inputs
        rgb_mid = data["rgb_mid"]
        rgb_seq = data["rgb_seq"]
        rgb_batch = data["rgb_batch"]
        prnu_in = data["prnu"]
        audio_in = data["audio"]
        
        with torch.no_grad():
            # A. Router Strategy
            weights = system.router(rgb_mid)
            w_temp, w_art, w_noise, w_audio = weights[0]
            
            # B. Experts Execution
            # Temporal
            temp_logits = system.expert_temp(rgb_seq)
            temp_timeline = torch.sigmoid(temp_logits).squeeze().cpu().numpy() 
            temp_score = float(temp_timeline.mean())

            # Artifacts
            art_batch_logits = system.expert_art(rgb_batch)
            art_masks = torch.sigmoid(art_batch_logits).squeeze(1).cpu().numpy()
            
            # Top 5 Frames for Display
            frame_scores = np.sum(art_masks > 0.5, axis=(1, 2))
            top_indices = np.argsort(frame_scores)[::-1][:5]
            art_score = float(art_masks.max())
            
            # Noise
            noise_logits = system.expert_noise_head(system.expert_noise_net(prnu_in))
            noise_score = float(torch.sigmoid(noise_logits).item())

            # Audio
            if audio_in.sum() == 0:
                print(">> No audio track detected.")
                audio_score = 0.5
            else:
                audio_logits = system.expert_audio(audio_in)
                audio_score = float(torch.sigmoid(audio_logits).item())

            # C. Verdict
            final_prob = (temp_score*w_temp) + (art_score*w_art) + (noise_score*w_noise) + (audio_score*w_audio)
            verdict_text = 'FAKE' if final_prob > 0.5 else 'REAL'

            # D. Visualization (Save to Disk)
            vis_frames = data["vis_frames"]
            vis_prnu   = prnu_in.squeeze().cpu().numpy()
            vis_audio  = data["vis_audio"]

            # Generate Report Filename
            report_filename = f"{Path(video_path).stem}_report.png"
            # Ensure we are saving to the same folder as the video (/app/uploads)
            report_path = os.path.join(os.path.dirname(video_path), report_filename)
            
            save_visual_report(
                report_path, 
                float(final_prob), 
                temp_timeline, 
                (top_indices, art_masks, vis_frames), 
                vis_prnu, 
                vis_audio
            )

        print(f"[INFO] Analysis Complete. Verdict: {verdict_text} ({final_prob:.2%})")

        return {
            "file": video_path,
            "prediction": verdict_text,
            "confidence": round(float(final_prob) * 100, 2),
            "breakdown": {
                "temporal": round(temp_score, 2),
                "artifact": round(art_score, 2),
                "noise": round(noise_score, 2),
                "audio": round(audio_score, 2)
            },
            "report_image": report_filename
        }

    except Exception as e:
        print(f"[ERROR] Processing Error: {e}")
        return {"status": "Failed", "error": str(e)}

def save_visual_report(save_path, verdict, timeline, artifact_data, prnu_map, audio_spec):
    print(f"[DEBUG] Generating report image at: {save_path}")
    
    try:
        import matplotlib.gridspec as gridspec
        
        top_indices, all_masks, all_frames = artifact_data
        
        fig = plt.figure(figsize=(18, 9), constrained_layout=True)
        plt.suptitle(f"Forensic Analysis | Verdict: {verdict:.1%} FAKE", fontsize=16, fontweight='bold')
        
        gs = gridspec.GridSpec(2, 4, figure=fig)

        # 1. Temporal
        ax1 = plt.subplot(gs[0, 0])
        ax1.plot(timeline, color='crimson', marker='o', markersize=4)
        ax1.set_title("Temporal Glitch")
        ax1.set_ylim(0, 1.1)
        ax1.grid(alpha=0.3)

        # 2. Noise
        ax2 = plt.subplot(gs[0, 1])
        ax2.imshow(prnu_map, cmap='gray')
        ax2.set_title("Camera Noise (PRNU)")
        ax2.axis('off')
        
        # 3. Audio
        ax3 = plt.subplot(gs[0, 2])
        ax3.imshow(audio_spec, cmap='inferno', origin='lower', aspect='auto')
        ax3.set_title("Audio Spectrum")
        ax3.axis('off')

        # 4. Confidence
        ax4 = plt.subplot(gs[0, 3])
        ax4.bar(["Real", "Fake"], [1-verdict, verdict], color=['#00cc66', '#ff3366'])
        ax4.set_title(f"Confidence: {verdict:.1%}")
        ax4.set_ylim(0, 1)
        
        # Row 2: Artifacts
        for i in range(4):
            if i >= len(top_indices): break
            idx = top_indices[i]
            ax = plt.subplot(gs[1, i])
            
            mask = all_masks[idx]
            frame = all_frames[idx]
            
            if np.sum(mask > 0.5) > 10:
                heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                overlay = cv2.addWeighted(np.uint8(255 * frame), 0.7, heatmap, 0.3, 0)
            else:
                overlay = np.uint8(255 * frame)

            ax.imshow(overlay)
            ax.set_title(f"Frame {idx}")
            ax.axis('off')

        # Save
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        # Verification Check
        if os.path.exists(save_path):
            print(f"[SUCCESS] Report saved successfully to disk.")
        else:
            print(f"[ERROR] Matplotlib finished but file is missing: {save_path}")

    except Exception as e:
        print(f"[CRITICAL ERROR] Matplotlib failed: {e}")