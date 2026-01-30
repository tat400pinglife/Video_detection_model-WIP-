import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import os
from tqdm import tqdm
import numpy as np

from model_architecture import MoE_Investigator

# CONFIGURATION
DATA_FOLDER = "./data/processed_data"
BATCH_SIZE = 16    # Keep this small for stability, increase to 32 later
LR = 0.0001        # Lower learning rate is safer, but slower
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DATASET
class ForensicDataset(Dataset):
    def __init__(self, folder_path):
        self.files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".pt"):
                    self.files.append(os.path.join(root, file))
        print(f"Dataset loaded: {len(self.files)} samples.")

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        try:
            path = self.files[idx]
            data = torch.load(path, weights_only=False)
            
            # Helper to squeeze dims and protect against NaNs
            def fix(t): 
                if t.ndim == 2: t = t.unsqueeze(0)
                if t.ndim == 4: t = t.squeeze(0)
                # NAN GUARD: Replace broken math with zeros
                return torch.nan_to_num(t.float(), nan=0.0)

            rgb   = data['rgb_mid'].float()
            diff  = fix(data['diff'])
            prnu  = fix(data['prnu'])
            fft   = fix(data['fft'])
            audio = fix(data['audio'])
            label = torch.tensor([data['label']], dtype=torch.float32)
            
            return rgb, diff, prnu, fft, audio, label
        except:
            # Return safe dummy data if file is corrupt
            return (torch.zeros(3,256,256), torch.zeros(1,256,256), torch.zeros(1,256,256), 
                    torch.zeros(1,256,256), torch.zeros(1,128,128), torch.tensor([0.0]))

# TRAINING LOOP
def train():
    print(f"Initializing STABLE MoE Training on {DEVICE}...")
    
    full_dataset = ForensicDataset(DATA_FOLDER)
    if len(full_dataset) == 0: return

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    model = MoE_Investigator(
        temp_path="models/temporal_model.pth",
        art_path="models/artifact_model.pth",
        noise_path="models/noise_model.pth",
        freq_path="models/frequency_model.pth",
        audio_path="models/audio_model.pth"
    ).to(DEVICE)

    # Use a slightly more robust optimizer setting
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        avg_weights = np.zeros(5) 
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch in loop:
            rgb, diff, prnu, fft, audio, labels = [x.to(DEVICE) for x in batch]
            
            # Skip empty batches
            if rgb.sum() == 0: continue

            optimizer.zero_grad()
            
            # Forward Pass
            predictions, route_weights = model(rgb, diff, prnu, fft, audio)
            loss = criterion(predictions, labels)
            
            if torch.isnan(loss):
                print("WARNING: Loss became NaN. Skipping batch to save model.")
                continue

            loss.backward()

            # gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            
            running_loss += loss.item()
            avg_weights += route_weights.detach().cpu().numpy().mean(axis=0)
            
        avg_weights /= len(train_loader)
        val_acc = evaluate(model, val_loader)
        
        # Visualize Weights
        w_str = " | ".join([f"{x*100:.1f}%" for x in avg_weights])
        
        print(f"E{epoch+1} Loss: {running_loss/len(train_loader):.3f} | Acc: {val_acc:.1f}%")
        print(f"Weights: [ Motn | Artf | Nois | Audi | Freq ]")
        print(f"         [ {w_str} ]")

        if val_acc >= best_acc:
            best_acc = val_acc
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/router_weights.pth")

def evaluate(model, loader):
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for batch in loader:
            rgb, diff, prnu, fft, audio, labels = [x.to(DEVICE) for x in batch]
            if rgb.sum() == 0: continue
            
            preds, _ = model(rgb, diff, prnu, fft, audio)
            
            # Use Sigmoid for binary classification
            predicted_labels = (torch.sigmoid(preds) > 0.5).float()
            correct += (predicted_labels == labels).sum().item()
            total += labels.size(0)
            
    return 100 * correct / (total + 1e-8)

if __name__ == "__main__":
    train()