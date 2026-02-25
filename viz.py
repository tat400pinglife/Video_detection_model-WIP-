import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
This is for purely visualizing the math metrics of videos
"""

def compute_radial_profile(z):
    """Calculates the 1D radial power spectrum from a 2D FFT map."""
    y, x = np.indices(z.shape)
    center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2).astype(int)
    tbin = np.bincount(r.ravel(), z.ravel())
    nr = np.bincount(r.ravel())
    return tbin / nr

def analyze_tensor_file(file_path):
    """Extracts 5 distinct 1D signals from a single saved .pt dictionary."""
    data = torch.load(file_path, map_location='cpu', weights_only=False)
    results = {}

    # 1. 1D Radial FFT Profile
    if 'fft' in data:
        fft_2d = data['fft'].squeeze().to(torch.float32).numpy()
        results['fft_1d'] = compute_radial_profile(fft_2d)

    if 'rgb_batch' in data:
        # Shape: [32, 3, 256, 256]. Convert to float32 [0.0 - 1.0]
        rgb = data['rgb_batch'].to(torch.float32).numpy() / 255.0
        
        # 2. Temporal Motion Jitter
        diff_seq = np.abs(rgb[1:] - rgb[:-1]) 
        motion_jitter = np.mean(diff_seq, axis=(1, 2, 3))
        results['motion_jitter'] = motion_jitter
        
        # 3. Green Channel Temporal Stability
        green_signal = np.mean(rgb[:, 1, :, :], axis=(1, 2))
        if np.std(green_signal) > 0:
            green_signal = (green_signal - np.mean(green_signal)) / np.std(green_signal)
        results['green_signal'] = green_signal

        # 4. Geometric Lighting Proxy (Left/Right Illumination Ratio)
        # Split width (256) in half. Left = 0:128, Right = 128:256
        left_brightness = np.mean(rgb[:, :, :, :128], axis=(1, 2, 3))
        right_brightness = np.mean(rgb[:, :, :, 128:], axis=(1, 2, 3))
        # Add small epsilon to prevent division by zero
        lighting_ratio = left_brightness / (right_brightness + 1e-6) 
        
        # Normalize to see the variance clearly
        if np.std(lighting_ratio) > 0:
            lighting_ratio = (lighting_ratio - np.mean(lighting_ratio)) / np.std(lighting_ratio)
        results['lighting_ratio'] = lighting_ratio

    # 5. Audio-Visual Kinematic Envelope (Proxy for AV Desync)
    if 'audio' in data and 'motion_jitter' in results:
        # Extract audio volume envelope across time (axis 1 in a 128x128 spectrogram)
        audio_spec = data['audio'].squeeze().to(torch.float32).numpy()
        audio_env = np.mean(audio_spec, axis=0) 
        
        # Normalize audio envelope
        if np.std(audio_env) > 0:
            audio_env = (audio_env - np.mean(audio_env)) / np.std(audio_env)
            
        # Interpolate motion jitter (length 31) to match audio envelope (length 128)
        x_motion = np.linspace(0, 1, len(results['motion_jitter']))
        x_audio = np.linspace(0, 1, len(audio_env))
        motion_env_stretched = np.interp(x_audio, x_motion, results['motion_jitter'])
        
        # Normalize stretched motion
        if np.std(motion_env_stretched) > 0:
            motion_env_stretched = (motion_env_stretched - np.mean(motion_env_stretched)) / np.std(motion_env_stretched)
            
        results['audio_env'] = audio_env
        results['motion_env'] = motion_env_stretched

    return results

def process_folder(folder_path):
    """Processes a folder and averages all the 1D signals."""
    files = [f for f in os.listdir(folder_path) if f.endswith('.pt')]
    sums = {}
    counts = {}

    print(f"Processing {len(files)} files in {folder_path}...")
    for f in tqdm(files, desc="Extracting Signals"):
        file_path = os.path.join(folder_path, f)
        try:
            signals = analyze_tensor_file(file_path)
            for key, signal in signals.items():
                if key not in sums:
                    sums[key] = np.zeros_like(signal)
                    counts[key] = 0
                
                min_len = min(len(sums[key]), len(signal))
                sums[key][:min_len] += signal[:min_len]
                counts[key] += 1
        except Exception:
            continue

    return {key: sums[key] / counts[key] for key in sums if counts[key] > 0}

import numpy as np
import scipy.stats as stats

def calculate_mathematical_metrics(data_dict, dataset_name="Dataset"):
    print(f"\nMathematical Analysis: {dataset_name} ---")
    
    # 1. Pearson Correlation (Audio-Visual Sync)
    if 'audio_env' in data_dict and 'motion_env' in data_dict:
        audio = data_dict['audio_env']
        motion = data_dict['motion_env']
        r, p_value = stats.pearsonr(audio, motion)
        print(f"1. Audio-Visual Correlation (Pearson r):  {r:+.4f} " 
              f"({'Sync' if r > 0.2 else 'Desync/Noise'})")
              
    # 2. Motion Jitter Metrics
    if 'motion_jitter' in data_dict:
        motion_seq = data_dict['motion_jitter']
        # Total Variation
        tv = np.sum(np.abs(np.diff(motion_seq)))
        # Variance of the first derivative
        derivative_var = np.var(np.diff(motion_seq))
        
        print(f"2. Motion Total Variation:                {tv:.4f}")
        print(f"   Motion Derivative Variance (Jitter):   {derivative_var:.6f}")

    # 3. High-Frequency Energy Ratio (FFT)
    if 'fft_1d' in data_dict:
        fft_seq = data_dict['fft_1d']
        total_bins = len(fft_seq)
        cutoff = int(total_bins * 0.75) # Top 25% of frequencies
        
        high_freq_energy = np.sum(fft_seq[cutoff:])
        total_energy = np.sum(fft_seq)
        hfer = high_freq_energy / (total_energy + 1e-8)
        
        print(f"3. High-Frequency Energy Ratio (HFER):    {hfer:.6f}")


def main():
    real_folder = "data/processed_data/real" 
    ai_folder = "data/processed_data/fake" 
    calculate_mathematical_metrics(process_folder(real_folder), "Real Videos")
    calculate_mathematical_metrics(process_folder(ai_folder), "AI Generated Videos")
    real_data = process_folder(real_folder)
    ai_data = process_folder(ai_folder)

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
    fig.suptitle("Visualization of frame data", fontsize=18, weight='bold')

    def plot_metric(ax, key, title, xlabel, ylabel, log_scale=False):
        if key in real_data and key in ai_data:
            ax.plot(real_data[key], label='Real', color='#1f77b4', linewidth=2)
            ax.plot(ai_data[key], label='AI Gen', color='#d62728', linewidth=2, linestyle='--')
            ax.set_title(title, weight='bold')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            if log_scale: ax.set_yscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)

    # Top Row: Spatial/Frequency Data
    plot_metric(axes[0, 0], 'fft_1d', "1. 1D Radial Power Spectrum", "Frequency", "Energy", log_scale=True)
    plot_metric(axes[0, 1], 'motion_jitter', "2. Temporal Motion Jitter", "Frame Transition", "Pixel Variance")
    plot_metric(axes[0, 2], 'green_signal', "3. Green Channel Stability", "Frame", "Z-Score")

    # Bottom Row: Advanced Proxies
    plot_metric(axes[1, 0], 'lighting_ratio', "4. Lighting Consistency Proxy (L/R Ratio)", "Frame", "Z-Score Ratio")
    
    # 5. Real Audio-Visual Sync Plot
    ax_real_av = axes[1, 1]
    if 'audio_env' in real_data and 'motion_env' in real_data:
        # Using Green and Blue to differentiate from the AI plot
        ax_real_av.plot(real_data['audio_env'], label='Real Audio Env', color='#2ca02c', linewidth=2) 
        ax_real_av.plot(real_data['motion_env'], label='Real Motion Env', color='#1f77b4', linewidth=2, linestyle='--') 
        ax_real_av.set_title("5. Real Audio-Visual Sync Proxy", weight='bold')
        ax_real_av.set_xlabel("Time (Normalized)")
        ax_real_av.set_ylabel("Z-Score Amplitude")
        ax_real_av.legend()
        ax_real_av.grid(True, alpha=0.3)

    # 6. AI Audio-Visual Desync Plot
    ax_ai_av = axes[1, 2]
    if 'audio_env' in ai_data and 'motion_env' in ai_data:
        ax_ai_av.plot(ai_data['audio_env'], label='AI Audio Env', color='purple', linewidth=2)
        ax_ai_av.plot(ai_data['motion_env'], label='AI Motion Env', color='orange', linewidth=2, linestyle='--')
        ax_ai_av.set_title("6. AI Audio-Visual Desync Proxy", weight='bold')
        ax_ai_av.set_xlabel("Time (Normalized)")
        ax_ai_av.set_ylabel("Z-Score Amplitude")
        ax_ai_av.legend()
        ax_ai_av.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.show()

    
if __name__ == "__main__":
    main()