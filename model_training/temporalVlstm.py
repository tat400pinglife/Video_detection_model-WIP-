import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import os
from pathlib import Path
import random

# Import your model
from model_architecture import TemporalDetector
import warnings
warnings.filterwarnings("ignore")

# CONFIG
DATA_FOLDER = "./data/processed_data"
SAVE_PATH = "models/temporal_lstm.pth"
BATCH_SIZE = 16  
SEQ_LEN = 10     
LR = 0.0001
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TemporalSequenceDataset(Dataset):
    def __init__(self, folder_path, seq_len=5, is_train=True):
        self.files = list(Path(folder_path).rglob("*.pt"))
        self.seq_len = seq_len
        self.is_train = is_train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            data = torch.load(self.files[idx])
            # Load the sequence: [31, 1, 256, 256]
            full_seq = data['diff_seq'].float()
            label = torch.tensor([data['label']], dtype=torch.float32)

            total_frames = full_seq.shape[0]
            # Safety: If video is too short, pad it
            if total_frames < self.seq_len:
                pad = torch.zeros(self.seq_len - total_frames, 1, 256, 256)
                full_seq = torch.cat([full_seq, pad], dim=0)
                total_frames = self.seq_len

            # SLICING STRATEGY
            if self.is_train:
                # Random slice for training (Augmentation)
                max_start = total_frames - self.seq_len
                start_idx = random.randint(0, max_start)
            else:
                # Center slice for validation (Consistency)
                start_idx = (total_frames - self.seq_len) // 2

            # Extract the clip
            clip = full_seq[start_idx : start_idx + self.seq_len]
            
            # Shape check: [Seq, 1, 256, 256]
            return clip, label

        except Exception as e:
            # print(f"Error loading {self.files[idx]}: {e}")
            return torch.zeros(self.seq_len, 1, 256, 256), torch.tensor([0.0])

def train():
    print(f"Using Device: {DEVICE}")
    
    # 1. Setup Data
    # We split the file list first to ensure no leakage
    all_files = list(Path(DATA_FOLDER).rglob("*.pt"))
    random.shuffle(all_files)
    
    split_idx = int(0.8 * len(all_files))
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    
    # Create Datasets
    # We pass the list of files to the dataset (requires modifying __init__ slightly or just filtering inside)
    # Ideally, we just point the dataset to the folder, but to handle train/val split correctly with random slicing:
    full_dataset = TemporalSequenceDataset(DATA_FOLDER, seq_len=SEQ_LEN, is_train=True)
    
    # Split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])
    
    # Turn off random slicing for validation set
    val_set.dataset.is_train = False 

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # 2. Setup Model
    model = TemporalDetector(sequence_length=SEQ_LEN).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    # 3. Training Loop
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
# --- VALIDATION STEP ---
        model.eval()
        correct = 0
        total = 0
        val_loss = 0
        
        print("\n[DEBUG] Validation Sample Predictions:")
        
        with torch.no_grad():
            batch_count = 0
            for batch in val_loader:
                # FIX: Unpack only 2 items (Input, Label)
                # The validation loader for Temporal only yields (diff_seq, label)
                diff_seq, labels = [x.to(DEVICE) for x in batch]
                
                # Forward Pass (Only feed diff_seq)
                output = model(diff_seq) 
                
                # Calculate Loss
                loss = criterion(output, labels)
                val_loss += loss.item()
                
                # Get Predictions
                preds = torch.sigmoid(output)

                # Accuracy Calc
                predicted_labels = (preds > 0.5).float()
                correct += (predicted_labels == labels).sum().item()
                total += labels.size(0)
                batch_count += 1
                
        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
        avg_val_loss = val_loss / len(val_loader)
        acc = 100 * correct / (total + 1e-8)
        
        print(f"Epoch {epoch+1} | Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Acc: {acc:.2f}%")
        # stop if loss is less 0.2
        if avg_train_loss < 0.2:
            torch.save(model.state_dict(), SAVE_PATH)
            print("Model saved.")
            break

    torch.save(model.state_dict(), SAVE_PATH)
    print("Done.")

if __name__ == "__main__":
    train()