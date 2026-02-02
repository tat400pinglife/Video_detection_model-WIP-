import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split
from pathlib import Path
import numpy as np
import warnings

# Import Architecture
from model_architecture import FrequencyExpert

# CONFIG
DATA_FOLDER = "./data/processed_data" 
SAVE_PATH = "models/frequency_model.pth"
BATCH_SIZE = 32
LR = 0.0001 
EPOCHS = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4 

warnings.filterwarnings("ignore")

class RobustFrequencyDataset(Dataset):
    def __init__(self, folder_path):
        # Scan for all .pt files
        self.files = list(Path(folder_path).rglob("*.pt"))
        print(f"Index complete. Found {len(self.files)} samples.")
        self.labels = []
        valid_files = []
        for f in self.files:
            if "real" in str(f.parent).lower():
                self.labels.append(0)
                valid_files.append(f)
            elif "fake" in str(f.parent).lower():
                self.labels.append(1)
                valid_files.append(f)
            else:
                pass 
        
        self.files = valid_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            path = self.files[idx]
            data = torch.load(path, weights_only=False)
            # Extract FFT Tensor
            if 'fft' not in data:
                return None
                
            x = data['fft'].float() # [1, 256, 256]
            
            # Ensure correct shape
            if x.ndim == 2: x = x.unsqueeze(0)
            if x.ndim == 4: x = x.squeeze(0)
            
            y = torch.tensor([data['label']], dtype=torch.float32)
            return x, y
        except Exception:
            # Return None so collate_fn can drop it
            return None

def drop_corrupt_collate(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0: return None, None
    
    inputs, labels = zip(*batch)
    return torch.stack(inputs), torch.stack(labels)

def train():
    print(f"Using device: {DEVICE}")
    
    # 1. Setup Data
    dataset = RobustFrequencyDataset(DATA_FOLDER)
    if len(dataset) == 0:
        print("Error: No data found.")
        return

    # 2. Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    
    # 3. Balance Classes (Train Set Only)
    train_indices = train_set.indices
    train_labels = [dataset.labels[i] for i in train_indices]
    
    class_counts = np.bincount(train_labels)
    if len(class_counts) < 2:
        print("Warning: Only one class detected. Disabling sampler.")
        sampler = None
    else:
        weights = 1. / (class_counts + 1e-6)
        samples_weights = [weights[l] for l in train_labels]
        sampler = WeightedRandomSampler(samples_weights, len(samples_weights))

    # 4. Loaders
    train_loader = DataLoader(
        train_set, 
        batch_size=BATCH_SIZE, 
        sampler=sampler, 
        num_workers=NUM_WORKERS,
        collate_fn=drop_corrupt_collate,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_set, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        collate_fn=drop_corrupt_collate,
        pin_memory=True
    )

    # 5. Model
    model = FrequencyExpert().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    # 6. Loop
    best_acc = 0.0
    print(f"Starting training on {len(train_set)} samples...")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        count = 0
        
        for X, y in train_loader:
            if X is None: continue # Skip empty batches
            
            X, y = X.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            count += 1
            
        avg_train_loss = train_loss / max(count, 1)

        # Validation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X, y in val_loader:
                if X is None: continue
                X, y = X.to(DEVICE), y.to(DEVICE)
                
                # Sigmoid for accuracy
                probs = torch.sigmoid(model(X))
                preds = (probs > 0.5).float()
                
                correct += (preds == y).sum().item()
                total += y.size(0)
        
        acc = 100 * correct / (total + 1e-8)
        print(f"Epoch {epoch+1} | Loss: {avg_train_loss:.4f} | Acc: {acc:.2f}%")
        
        if acc >= best_acc and acc > 50.0: # Only save if better than random guessing
            best_acc = acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f">> Saved Best Model ({acc:.2f}%)")

    print(f"Done. Final model saved to {SAVE_PATH}")

if __name__ == "__main__":
    train()