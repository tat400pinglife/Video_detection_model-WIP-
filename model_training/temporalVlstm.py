import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
import random
import numpy as np

# Import your model
from model_architecture import TemporalDetector
import warnings
warnings.filterwarnings("ignore")

# CONFIG
DATA_FOLDER = "./data/processed_data"
SAVE_PATH = "models/temporal_lstm.pth"
BATCH_SIZE = 16 # unless you have a giga device dont increase this by too much
SEQ_LEN = 12     
LR = 0.0001
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TemporalSequenceDataset(Dataset):
    def __init__(self, file_list, seq_len=12, is_train=True):
        self.files = file_list
        self.seq_len = seq_len
        self.is_train = is_train
        print(f"Temporal Dataset ({'Train' if is_train else 'Val'}): Found {len(self.files)} samples.")
        self.weights = torch.tensor([0.299, 0.587, 0.114]).view(3, 1, 1)

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        try:
            path = self.files[idx]
            data = torch.load(path, weights_only=False)
            
            rgb = data['rgb_batch'].float() / 255.0 
            gray = (rgb * self.weights.to(rgb.device)).sum(dim=1)
            
            # --- FEATURE 1: MOTION METRICS ---
            diff_seq_raw = torch.abs(gray[1:] - gray[:-1]) # [31, 256, 256]
            
            # Average spatially to get a 1D motion signal: [31]
            motion_1d = diff_seq_raw.mean(dim=(1, 2)).numpy()
            
            tv = np.sum(np.abs(np.diff(motion_1d)))
            jitter = np.var(np.diff(motion_1d))
            
            # --- FEATURE 2: HFER METRIC ---
            fft_2d = data['fft'].squeeze().float().numpy() # [256, 256]
            y_idx, x_idx = np.indices(fft_2d.shape)
            center = np.array([(x_idx.max()-x_idx.min())/2.0, (y_idx.max()-y_idx.min())/2.0])
            r = np.sqrt((x_idx - center[0])**2 + (y_idx - center[1])**2).astype(int)
            
            tbin = np.bincount(r.ravel(), fft_2d.ravel())
            nr = np.bincount(r.ravel())
            radial_profile = tbin / nr
            
            cutoff = int(len(radial_profile) * 0.75)
            hfer = np.sum(radial_profile[cutoff:]) / (np.sum(radial_profile) + 1e-8)
            
            # Pack the 3 scalars into a single tabular tensor
            tabular_features = torch.tensor([tv, jitter, hfer], dtype=torch.float32)

            # --- PREPARE SEQUENCE FOR MODEL ---
            diff_seq = diff_seq_raw.unsqueeze(1) # [31, 1, 256, 256]
            total_frames = diff_seq.shape[0]
            
            if total_frames > self.seq_len:
                if self.is_train:
                    start_idx = random.randint(0, total_frames - self.seq_len)
                else:
                    start_idx = (total_frames - self.seq_len) // 2
                diff_seq = diff_seq[start_idx : start_idx + self.seq_len]
            elif total_frames < self.seq_len:
                pad_size = self.seq_len - total_frames
                pad = torch.zeros((pad_size, 1, 256, 256), dtype=diff_seq.dtype)
                diff_seq = torch.cat([diff_seq, pad], dim=0)

            y = torch.tensor([data['label']], dtype=torch.float32)
            
            # Return sequence, the new scalar features, and the label
            return diff_seq, tabular_features, y

        except Exception as e:
            new_idx = random.randint(0, len(self.files) - 1)
            return self.__getitem__(new_idx)
def train():
    print(f"Using Device: {DEVICE}")
    
    # 1. Setup Data
    all_files = list(Path(DATA_FOLDER).rglob("*.pt"))
    if not all_files:
        print("No files found! Check DATA_FOLDER.")
        return
        
    random.shuffle(all_files)
    
    split_idx = int(0.8 * len(all_files))
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    
    train_set = TemporalSequenceDataset(train_files, seq_len=SEQ_LEN, is_train=True)
    val_set = TemporalSequenceDataset(val_files, seq_len=SEQ_LEN, is_train=False)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # 2. Setup Model
    model = TemporalDetector(sequence_length=SEQ_LEN).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    # 3. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            # --- THE FIX: Unpack 3 items now (Sequence, Math Scalars, Labels) ---
            X_seq, X_tab, y = [item.to(DEVICE) for item in batch]
            
            optimizer.zero_grad()
            
            # Pass both the visual sequence and the tabular math to the model
            pred = model(X_seq, X_tab)
            loss = criterion(pred, y)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # VALIDATION STEP
        model.eval()
        correct = 0
        total = 0
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # --- THE FIX: Unpack 3 items in validation as well ---
                X_seq, X_tab, labels = [item.to(DEVICE) for item in batch]
                
                # Pass both to the model
                output = model(X_seq, X_tab) 
                
                loss = criterion(output, labels)
                val_loss += loss.item()
                
                preds = torch.sigmoid(output)
                predicted_labels = (preds > 0.5).float()
                correct += (predicted_labels == labels).sum().item()
                total += labels.size(0)
                
        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        acc = 100 * correct / (total + 1e-8)
        
        print(f"Epoch {epoch+1} | Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Acc: {acc:.2f}%")
        
        # Stop early if the loss drops below 0.2
        if avg_train_loss < 0.2:
            torch.save(model.state_dict(), SAVE_PATH)
            print("Model saved due to low training loss.")
            break

    # Save final model if it didn't hit early stopping
    torch.save(model.state_dict(), SAVE_PATH)
    print("Done.")

if __name__ == "__main__":
    train()