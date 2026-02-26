"""
Docstring for model_training.check

This is a quick check on the accuracy for each model. Will be used to find signs of over/underfitting.

"""


import torch
from torch.utils.data import DataLoader
from moe2 import ForensicDataset, DATA_FOLDER, SEQ_LEN
from model_architecture import MoE_Investigator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def check():
    # Load Data
    ds = ForensicDataset(DATA_FOLDER, seq_len=SEQ_LEN)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    
    # Load Model
    model = MoE_Investigator(
        temp_path="models/temporal_lstm.pth",
        art_path="models/artifact_model.pth",
        noise_path="models/noise_model.pth",
        freq_path="models/frequency_model.pth",
        audio_path="models/audio_model.pth"
    ).to(DEVICE)
    model.eval()
    
    print(f"\n--- Expert Check ({len(ds)} samples) ---")
    
    accs = {"temp": 0, "art": 0, "noise": 0, "freq": 0, "audio": 0, "moe": 0}
    
    with torch.no_grad():
        for batch in loader:
            rgb, diff_seq, prnu, fft, audio, label = [x.to(DEVICE) for x in batch]
            if rgb.sum() == 0: continue
            
            # Run manually to see individual outputs
            p_temp = torch.sigmoid(model.expert_temp(diff_seq))
            p_art  = torch.sigmoid(model.expert_art(rgb))
            p_art  = p_art.flatten(1).max(1)[0].unsqueeze(1) # Fix shape
            p_noise= torch.sigmoid(model.expert_noise_head(model.expert_noise_net(prnu)))
            p_freq = torch.sigmoid(model.expert_freq(fft))
            p_audio= torch.sigmoid(model.expert_audio(audio))
            
            # MoE Prediction
            p_moe, _ = model(rgb, diff_seq, prnu, fft, audio)
            
            # Check accuracy for each
            lbl = label.item()
            if (p_temp.item() > 0.5) == (lbl == 1.0): accs["temp"] += 1
            if (p_art.item()  > 0.5) == (lbl == 1.0): accs["art"]  += 1
            if (p_noise.item()> 0.5) == (lbl == 1.0): accs["noise"]+= 1
            if (p_freq.item() > 0.5) == (lbl == 1.0): accs["freq"] += 1
            if (p_audio.item()> 0.5) == (lbl == 1.0): accs["audio"]+= 1
            if (p_moe.item()  > 0.5) == (lbl == 1.0): accs["moe"]  += 1

    total = len(ds)
    print(f"Temporal Acc: {accs['temp']/total:.2%}")
    print(f"Artifact Acc: {accs['art']/total:.2%}")
    print(f"Noise Acc:    {accs['noise']/total:.2%}")
    print(f"Freq Acc:     {accs['freq']/total:.2%}")
    print(f"Audio Acc:    {accs['audio']/total:.2%}")
    print(f"----------------")
    print(f"MoE System:   {accs['moe']/total:.2%}")

if __name__ == "__main__":
    check()