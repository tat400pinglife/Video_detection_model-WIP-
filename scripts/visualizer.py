import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import random

def load_random_sample(folder):
    files = list(Path(f"data/frames/{folder}").glob("*.pt"))
    if not files: return None, None
    path = random.choice(files)
    data = torch.load(path)
    # Get middle frame (index 8)
    return data, path.name

def show_comparison():
    real_data, real_name = load_random_sample("real")
    fake_data, fake_name = load_random_sample("fake")
    
    if not real_data or not fake_data:
        print("Error: Could not find processed files. Run process_data.py first.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # ROW 1: REAL
    # RGB
    rgb_real = real_data['rgb'][8].permute(1, 2, 0).numpy()
    axes[0,0].imshow(np.clip(rgb_real, 0, 1))
    axes[0,0].set_title(f"REAL: {real_name}\n(RGB Frame 8)")
    axes[0,0].axis('off')
    
    # Diff
    diff_real = real_data['diff'][8].squeeze().numpy()
    axes[0,1].imshow(diff_real, cmap='gray')
    axes[0,1].set_title("Temporal Diff")
    axes[0,1].axis('off')

    # FFT
    fft_real = real_data['fft'][8].squeeze().numpy()
    axes[0,2].imshow(fft_real, cmap='inferno')
    axes[0,2].set_title("Spatial FFT")
    axes[0,2].axis('off')

    # ROW 2: FAKE
    # RGB
    rgb_fake = fake_data['rgb'][8].permute(1, 2, 0).numpy()
    axes[1,0].imshow(np.clip(rgb_fake, 0, 1))
    axes[1,0].set_title(f"FAKE: {fake_name}\n(RGB Frame 8)")
    axes[1,0].axis('off')

    # Diff
    diff_fake = fake_data['diff'][8].squeeze().numpy()
    axes[1,1].imshow(diff_fake, cmap='gray')
    axes[1,1].set_title("Temporal Diff")
    axes[1,1].axis('off')

    # FFT
    fft_fake = fake_data['fft'][8].squeeze().numpy()
    axes[1,2].imshow(fft_fake, cmap='inferno')
    axes[1,2].set_title("Spatial FFT")
    axes[1,2].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    show_comparison()