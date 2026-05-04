import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
import warnings

warnings.filterwarnings("ignore")

def load_data_sample(data_dir, max_samples=50):
    """Loads a random subset of tensors to prevent RAM overflow during analysis."""
    base_path = Path(data_dir)
    data_stats = {
        "real": {"motion": [], "prnu": [], "fft": [], "audio": []},
        "fake": {"motion": [], "prnu": [], "fft": [], "audio": []}
    }
    
    # Store a few raw 2D maps to calculate the "Average Visual Artifact"
    avg_maps = {
        "real": {"prnu": [], "fft": []},
        "fake": {"prnu": [], "fft": []}
    }
    
    weights = torch.tensor([0.299, 0.587, 0.114]).view(3, 1, 1)

    for class_name in ["real", "fake"]:
        folder = base_path / class_name
        if not folder.exists(): continue
            
        files = list(folder.glob("*.pt"))
        # Shuffle and limit to max_samples for fast analysis
        random.shuffle(files)
        sample_files = files[:max_samples]
        
        print(f"Analyzing {len(sample_files)} {class_name} samples...")
        
        for file in sample_files:
            try:
                data = torch.load(file, weights_only=False)
                
                # 1. Motion 
                rgb = data['rgb_batch'].float() / 255.0
                gray = (rgb * weights.to(rgb.device)).sum(dim=1)
                motion_val = torch.abs(gray[1:] - gray[:-1]).mean().item()
                data_stats[class_name]["motion"].append(motion_val)
                
                # 2. PRNU
                prnu_tensor = data['prnu'].float()
                data_stats[class_name]["prnu"].append(prnu_tensor.abs().mean().item())
                avg_maps[class_name]["prnu"].append(prnu_tensor.squeeze().numpy())
                
                # 3. FFT
                fft_tensor = data['fft'].float()
                data_stats[class_name]["fft"].append(fft_tensor.mean().item())
                avg_maps[class_name]["fft"].append(fft_tensor.squeeze().numpy())
                
                # 4. Audio
                audio_val = data['audio'].float().mean().item()
                if audio_val > 0:
                    data_stats[class_name]["audio"].append(audio_val)
                    
            except Exception as e:
                continue
                
    return data_stats, avg_maps

def plot_minimal_dashboard(data_stats, avg_maps):
    """Generates a clean, professional visualization dashboard with minimal data ink."""
    fig = plt.figure(figsize=(16, 10))
    fig.canvas.manager.set_window_title("Dataset Feature Analysis")
    
    features = ["motion", "prnu", "fft", "audio"]
    titles = ["Temporal Motion Energy", "PRNU Noise Magnitude", "FFT Frequency Structure", "Audio Spectrogram Power"]
    
    # --- Row 1: Statistical Distributions (Boxplots) ---
    for i, (feat, title) in enumerate(zip(features, titles)):
        ax = plt.subplot(2, 4, i + 1)
        
        real_data = data_stats["real"][feat]
        fake_data = data_stats["fake"][feat]
        
        if not real_data or not fake_data:
            continue
            
        # Clean boxplot
        bp = ax.boxplot([real_data, fake_data], patch_artist=True, widths=0.5)
        
        # Styling
        colors = ['#2ca02c', '#d62728'] # Muted Green (Real), Muted Red (Fake)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
            patch.set_linewidth(0)
            
        for median in bp['medians']:
            median.set(color='black', linewidth=1.5)
            
        ax.set_xticklabels(['Real', 'Fake'], fontsize=11)
        ax.set_title(title, fontsize=12, pad=15)
        
        # Minimal data ink: Remove top and right spines, remove tick marks
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cccccc')
        ax.spines['bottom'].set_color('#cccccc')
        ax.tick_params(axis='both', which='both', length=0)
        ax.grid(axis='y', linestyle='--', alpha=0.3)

    # --- Row 2: 2D Spatial Averages (Visualizing the Artifacts) ---
    # We average the 2D maps across the sample to reveal structural deepfake patterns
    
    maps_to_plot = [
        ("Real PRNU", np.mean(avg_maps["real"]["prnu"], axis=0), "gray"),
        ("Fake PRNU", np.mean(avg_maps["fake"]["prnu"], axis=0), "gray"),
        ("Real FFT", np.mean(avg_maps["real"]["fft"], axis=0), "viridis"),
        ("Fake FFT", np.mean(avg_maps["fake"]["fft"], axis=0), "viridis")
    ]
    
    for i, (title, map_data, cmap) in enumerate(maps_to_plot):
        ax = plt.subplot(2, 4, i + 5)
        im = ax.imshow(map_data, cmap=cmap, aspect='auto')
        ax.set_title(title, fontsize=12, pad=10)
        
        # Minimal data ink: Strip all axes for image data
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).outline.set_visible(False)

    plt.tight_layout(pad=3.0)
    plt.savefig("feature_analysis_dashboard.png", dpi=300, bbox_inches='tight')
    print("Analysis complete. Dashboard saved to 'feature_analysis_dashboard.png'")

if __name__ == "__main__":
    dataset_path = "data/processed_data" 
    stats, maps = load_data_sample(dataset_path, max_samples=1000)
    plot_minimal_dashboard(stats, maps)