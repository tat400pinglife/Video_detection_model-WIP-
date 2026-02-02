import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import os
from tqdm import tqdm
import numpy as np

# Import the architecture
from model_architecture import MoE_Investigator

# CONFIGURATION
DATA_FOLDER = "./data/processed_data"
BATCH_SIZE = 16    
LR = 0.001        
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 5 # match with lstm model

# DATASET
class ForensicDataset(Dataset):
    def __init__(self, folder_path, seq_len=5):
        self.seq_len = seq_len
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
                return torch.nan_to_num(t.float(), nan=0.0)

            # 1. Load Standard Features
            rgb   = data['rgb_mid'].float()
            prnu  = fix(data['prnu'])
            fft   = fix(data['fft'])
            audio = fix(data['audio'])
            label = torch.tensor([data['label']], dtype=torch.float32)
            
            # 2. Load and Slice Sequence (New Logic)
            # Input shape in file: [31, 1, 256, 256]
            full_seq = data['diff_seq'].float()
            
            # Safety Pad
            if full_seq.shape[0] < self.seq_len:
                pad = torch.zeros(self.seq_len - full_seq.shape[0], 1, 256, 256)
                full_seq = torch.cat([full_seq, pad], dim=0)

            # Center Slice (Consistency for MoE)
            mid = full_seq.shape[0] // 2
            start = max(0, mid - (self.seq_len // 2))
            # Ensure we don't go out of bounds
            if start + self.seq_len > full_seq.shape[0]: start = 0
            
            diff_seq = full_seq[start : start + self.seq_len] 
            # Output shape: [5, 1, 256, 256]

            return rgb, diff_seq, prnu, fft, audio, label
            
        except Exception as e:
            # print(f"Error loading {path}: {e}")
            # Return safe dummy data
            dummy_seq = torch.zeros(self.seq_len, 1, 256, 256)
            return (torch.zeros(3,256,256), dummy_seq, torch.zeros(1,256,256), 
                    torch.zeros(1,256,256), torch.zeros(1,128,128), torch.tensor([0.0]))

# TRAINING LOOP
def train():
    print(f"Initializing STABLE MoE Training on {DEVICE}...")
    
    full_dataset = ForensicDataset(DATA_FOLDER, seq_len=SEQ_LEN)
    if len(full_dataset) == 0: return

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize MoE with the NEW LSTM path
    model = MoE_Investigator(
        temp_path="models/temporal_lstm.pth",
        art_path="models/artifact_model.pth",
        noise_path="models/noise_model.pth",
        freq_path="models/frequency_model.pth",
        audio_path="models/audio_model.pth"
    ).to(DEVICE)

    # Optimizer (Freeze experts, train router only?) 
    # Usually standard AdamW is fine if experts are frozen inside the class
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    #criterion = nn.BCEWithLogitsLoss()
    criterion = nn.BCELoss()
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        avg_weights = np.zeros(5) 
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch in loop:
            # Note: diff_seq is the second item now
            rgb, diff_seq, prnu, fft, audio, labels = [x.to(DEVICE) for x in batch]
            
            if rgb.sum() == 0: continue

            optimizer.zero_grad()
            
            # Forward Pass with Sequence
            predictions, route_weights = model(rgb, diff_seq, prnu, fft, audio)
            loss = criterion(predictions, labels)
            
            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            avg_weights += route_weights.detach().cpu().numpy().mean(axis=0)
            
        avg_weights /= len(train_loader)
        val_acc = evaluate(model, val_loader)
        
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
            rgb, diff_seq, prnu, fft, audio, labels = [x.to(DEVICE) for x in batch]
            if rgb.sum() == 0: continue
            
            preds, _ = model(rgb, diff_seq, prnu, fft, audio)
            
            predicted_labels = (torch.sigmoid(preds) > 0.5).float()
            correct += (predicted_labels == labels).sum().item()
            total += labels.size(0)
            
    return 100 * correct / (total + 1e-8)

if __name__ == "__main__":
    train()