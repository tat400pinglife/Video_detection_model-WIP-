import torch
from torch.utils.data import DataLoader
from moe import ForensicDataset, MoE_Investigator # Import from your existing files

# CONFIG
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_FOLDER = "./data/processed_data"

def inspect_experts():
    print(f"--- INSPECTING EXPERT OPINIONS on {DEVICE} ---")
    
    # 1. Load Data
    dataset = ForensicDataset(DATA_FOLDER)
    if len(dataset) == 0: return
    loader = DataLoader(dataset, batch_size=5, shuffle=True) # increase batch size when get big data
    
    # 2. Load Model
    model = MoE_Investigator(
        temp_path="models/temporal_model.pth",
        art_path="models/artifact_model.pth",
        noise_path="models/noise_model.pth",
        freq_path="models/frequency_model.pth",
        audio_path="models/audio_model.pth"
    ).to(DEVICE)
    model.eval()
    
    # 3. Get One Batch
    rgb, diff, prnu, fft, audio, labels = next(iter(loader))
    rgb, diff, prnu, fft, audio = [x.to(DEVICE) for x in [rgb, diff, prnu, fft, audio]]
    
    print(f"\n[Real Labels]: {labels.flatten().tolist()} (0=Real, 1=Fake)")
    print("-" * 60)
    print(f"{'TYPE':<10} | {'V1':<6} | {'V2':<6} | {'V3':<6} | {'V4':<6} | {'V5':<6}")
    print("-" * 60)

    with torch.no_grad():
        # RUN EXPERTS MANUALLY
        
        # 1. Temporal
        o_temp = torch.sigmoid(model.expert_temp(diff)).flatten().cpu().numpy()
        print(f"{'Motion':<10} | {o_temp[0]:.2f}   | {o_temp[1]:.2f}   | {o_temp[2]:.2f}   | {o_temp[3]:.2f}   | {o_temp[4]:.2f}")

        # 2. Artifacts (Max Glitch)
        o_art = torch.sigmoid(model.expert_art(rgb)).flatten(1).max(1)[0].cpu().numpy()
        print(f"{'Artifact':<10} | {o_art[0]:.2f}   | {o_art[1]:.2f}   | {o_art[2]:.2f}   | {o_art[3]:.2f}   | {o_art[4]:.2f}")

        # 3. Noise
        o_noise = torch.sigmoid(model.expert_noise_head(model.expert_noise_net(prnu))).flatten().cpu().numpy()
        print(f"{'Noise':<10} | {o_noise[0]:.2f}   | {o_noise[1]:.2f}   | {o_noise[2]:.2f}   | {o_noise[3]:.2f}   | {o_noise[4]:.2f}")

        # 4. Frequency
        o_freq = torch.sigmoid(model.expert_freq(fft)).flatten().cpu().numpy()
        print(f"{'Freq':<10} | {o_freq[0]:.2f}   | {o_freq[1]:.2f}   | {o_freq[2]:.2f}   | {o_freq[3]:.2f}   | {o_freq[4]:.2f}")

        # 5. Audio
        # Handle silence
        if audio.sum() == 0: o_audio = [0.5]*5
        else: o_audio = torch.sigmoid(model.expert_audio(audio)).flatten().cpu().numpy()
        print(f"{'Audio':<10} | {o_audio[0]:.2f}   | {o_audio[1]:.2f}   | {o_audio[2]:.2f}   | {o_audio[3]:.2f}   | {o_audio[4]:.2f}")

    print("-" * 60)
    print("INTERPRETATION:")
    print("If you see mostly 0.50 -> The Expert is untrained / dead.")
    print("If you see 0.00 or 1.00 but wrong -> The Expert is overconfident/broken.")
    print("If you see mostly correct values -> The Expert is GOOD, but the Router is dumb.")

if __name__ == "__main__":
    inspect_experts()