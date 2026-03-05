import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split
from pathlib import Path
import numpy as np
import warnings
import matplotlib.pyplot as plt

from model_architecture import AudioExpert

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore") # Ignore specific torch warnings if needed

# class AudioDataset(Dataset):
#     def __init__(self, root_dir):
#         # FAST INIT: We don't load files. We just scan paths.
#         self.files = list(Path(root_dir).rglob("*.pt"))
#         self.labels = []
        
#         print(f"Indexing {len(self.files)} files...")
        
#         valid_files = []
#         for f in self.files:
#             # Check path string for label
#             # Note: This relies on your create_tensors.py structure
#             if "real" in str(f.parent).lower():
#                 self.labels.append(0)
#                 valid_files.append(f)
#             elif "fake" in str(f.parent).lower():
#                 self.labels.append(1)
#                 valid_files.append(f)
#             else:
#                 # If structure is flat, we might have to load, but let's assume structure exists
#                 # Fallback: Check file name or skip
#                 pass 
                
#         self.files = valid_files
#         print(f"Index complete. Found {len(self.files)} samples.")

#     def __len__(self): return len(self.files)

#     def __getitem__(self, idx):
#         try:
#             path = self.files[idx]
#             data = torch.load(path, weights_only=False)
            
#             spec = data['audio'] # [1, 128, 128]
            
#             # --- SILENCE CHECK (Lazy) ---
#             # If audio is silent, return None. 
#             # The collate_fn will throw this away so it doesn't break the batch.
#             if spec.max() < 0.01:
#                 return None
            
#             # Ensure shape [1, 128, 128]
#             if spec.ndim == 2: spec = spec.unsqueeze(0)
#             if spec.ndim == 4: spec = spec.squeeze(0)
            
#             label = torch.tensor([data['label']], dtype=torch.float32)
            
#             return spec.float(), label
            
#         except Exception as e:
#             # On file corruption, return None
#             return None

class AudioDataset(Dataset):
    def __init__(self, folder_path):
        self.files = list(Path(folder_path).rglob("*.pt"))
        print(f"Audio Dataset: Found {len(self.files)} samples.")
                # Fast Label Scan for Class Balancing
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
                pass # Skip unknown folders

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        try:
            path = self.files[idx]
            data = torch.load(path, weights_only=False)
            
            # 1. Check for key (Safeguard)
            if 'audio' not in data: return None
                
            # 2. DECOMPRESS: Float16 -> Float32
            x = data['audio'].float() # [1, 256, 256]
            
            # 3. Shape Safety
            if x.ndim == 2: x = x.unsqueeze(0)
            
            y = torch.tensor([data['label']], dtype=torch.float32)
            return x, y
            
        except Exception:
            return None

# CUSTOM BATCH BUILDER
def drop_silence_collate(batch):
    """
    Filters out 'None' samples (silent audio) from the batch.
    """
    # Remove Nones
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        return None, None
        
    specs, labels = zip(*batch)
    return torch.stack(specs), torch.stack(labels)

def train():
    # 1. Setup Data
    full_dataset = AudioDataset("./data/processed_data")
    if len(full_dataset) == 0: 
        print("No data found. Check paths.")
        return

    # 2. Split Train/Val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    # 3. Handle Class Imbalance (For Train Set Only)
    # We need to extract labels from the Subset
    train_indices = train_set.indices
    train_labels = [full_dataset.labels[i] for i in train_indices]
    
    class_counts = np.bincount(train_labels)
    # Safety check for single-class batches
    if len(class_counts) < 2:
        print("Warning: Training set only has one class? Sampler disabled.")
        sampler = None
    else:
        weights = 1. / (class_counts + 1e-6) # Add epsilon
        samples_weights = [weights[l] for l in train_labels]
        sampler = WeightedRandomSampler(samples_weights, len(samples_weights))

    # 4. Loaders
    train_loader = DataLoader(
        train_set, 
        batch_size=32, 
        sampler=sampler, # Balances the training
        collate_fn=drop_silence_collate,
        num_workers=0 # Set to 2 or 4 on Linux/Mac
    )
    
    # Val loader doesn't need sampler, just shuffle=False
    val_loader = DataLoader(
        val_set, 
        batch_size=32, 
        shuffle=False,
        collate_fn=drop_silence_collate
    )

    # 5. Model
    model = AudioExpert().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.BCEWithLogitsLoss()
    
    print(f"Starting Training (Train: {train_size}, Val: {val_size})...")
    
    best_val_acc = 100.0
    patience = 5
    trigger_times = 0
    
    train_loss_data = []
    val_loss_data = []
    val_acc_data = []
    training_acc_data = []
    
    for epoch in range(100):
        # TRAIN
        model.train()
        train_loss = 0
        
        for specs, labels in train_loader:
            if specs is None: continue # Skip empty batches (all silent)
            
            specs, labels = specs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            out = model(specs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            # training acc
            train_acc = (torch.sigmoid(out) > 0.5).float()
            
        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
        training_acc_data.append(int(train_acc.sum().item()) / (len(train_loader.dataset) + 1e-8) * 100)
        
        # VALIDATE
        model.eval()
        correct = 0
        total = 0
        val_loss = 0
        
        with torch.no_grad():
            for specs, labels in val_loader:
                if specs is None: continue
                specs, labels = specs.to(DEVICE), labels.to(DEVICE)
                
                out = model(specs)
                loss = criterion(out, labels)
                val_loss += loss.item()
                
                preds = (torch.sigmoid(out) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        # Metrics
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_acc = 100 * correct / (total + 1e-8)
        
        val_loss_data.append(avg_val_loss)
        val_acc_data.append(val_acc)

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f} | Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "models/audio_expert.pth")
            print("Model saved.")
        
        else: 
            print("No improvement this epoch.")
            trigger_times += 1
            
            if trigger_times >= patience:
                print("Early Stopping!")
                break
    
    plt.figure(figsize=(10,5))
    plt.plot(train_loss_data, label='Training Loss')
    plt.plot(val_loss_data, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("audio_model_training_loss.png")
    
    plt.figure(figsize=(10,5))
    plt.plot(training_acc_data, label='Training Accuracy')
    plt.plot(val_acc_data, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig("audio_model_training_accuracy.png")
    print("Training Complete.")

if __name__ == "__main__":
    train()