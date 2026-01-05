import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import random

def load_random_sample(folder):
    files = list(Path(f"data/frames/{folder}").glob("*.pt"))
    if not files: return None, None
    path = random.choice(files)
    data = torch.load(path, weights_only=True)
    # Get middle frame (index 8)
    return data, path.name

def show_comparison():
    real_data, real_name = load_random_sample("real")
    fake_data, fake_name = load_random_sample("fake")
    
    if not real_data or not fake_data:
        print("Error: Could not find processed files. Run frame_extraction.py first.")
        return

    # UPDATED: 2 Rows, 4 Columns (Added PRNU)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # --- HELPER TO PLOT ---
    def plot_row(row_idx, data, name, label_type):
        # 1. RGB
        rgb = data['rgb'][8].permute(1, 2, 0).numpy()
        axes[row_idx, 0].imshow(np.clip(rgb, 0, 1))
        axes[row_idx, 0].set_title(f"{label_type}: {name}\n(RGB Frame 8)")
        axes[row_idx, 0].axis('off')

        # 2. Temporal Difference
        diff = data['diff'][8].squeeze().numpy()
        axes[row_idx, 1].imshow(diff, cmap='gray')
        axes[row_idx, 1].set_title("Temporal Diff")
        axes[row_idx, 1].axis('off')

        # 3. Spatial FFT
        fft = data['fft'][8].squeeze().numpy()
        axes[row_idx, 2].imshow(fft, cmap='inferno')
        axes[row_idx, 2].set_title("Spatial FFT")
        axes[row_idx, 2].axis('off')

        # 4. PRNU (Noise Residuals)
        if 'prnu' in data:
            prnu = data['prnu'][8].squeeze().numpy()
            
            axes[row_idx, 3].imshow(prnu, cmap='gray')
            axes[row_idx, 3].set_title("PRNU (Noise Residual)")
            axes[row_idx, 3].axis('off')
        else:
            axes[row_idx, 3].text(0.5, 0.5, "No PRNU Data", ha='center')

    # Plot Real
    plot_row(0, real_data, real_name, "REAL")
    
    # Plot Fake
    plot_row(1, fake_data, fake_name, "FAKE")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    show_comparison()