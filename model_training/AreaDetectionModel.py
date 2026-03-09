import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import random
import numpy as np
from model_architecture import ArtifactSegmentor
import matplotlib.pyplot as plt

# CONFIG
DATA_PATH = "./data/processed_data"
SAVE_PATH = "./models/artifact_model.pth"
BATCH_SIZE = 16
LR = 0.0001
EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SupervisedArtifactDataset(Dataset):
    def __init__(self, root_dir):
        self.files = list(Path(root_dir).rglob("*.pt"))
        print(f"Artifact Dataset: Found {len(self.files)} samples.")

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        try:
            path = self.files[idx]
            data = torch.load(path, weights_only=False)
            
            # 1. LOAD COMPRESSED RGB [32, 3, 256, 256] (Uint8)
            if 'rgb_batch' in data:
                # Pick 1 random frame from the batch
                ridx = random.randint(0, data['rgb_batch'].shape[0]-1)
                img_uint8 = data['rgb_batch'][ridx] # Shape: [3, 256, 256]
                
                # 2. DECOMPRESS: Uint8 (0-255) -> Float32 (0.0-1.0)
                img = img_uint8.float() / 255.0
            else:
                # Fallback for old files (if any)
                img = data['rgb_mid'].float()

            # 3. Create Label/Target
            label = float(data['label']) # 1.0 or 0.0
            
            # Create a segmentation mask (1.0 for Fake, 0.0 for Real)
            # This is a "Weakly Supervised" approach
            target = torch.full((1, 256, 256), label, dtype=torch.float32)
            
            return img, target
            
        except Exception as e:
            print(f"[WARN] Failed to load sample {idx}: {e}")
            return None

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None, None
    imgs, targets = zip(*batch)
    return torch.stack(imgs), torch.stack(targets)


def train_supervised():
    dataset = SupervisedArtifactDataset(DATA_PATH)
    
    # Split
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn)

    model = ArtifactSegmentor().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    patience = 5 # adjust accordingly
    trigger_times = 0

    train_loss_data = []
    val_loss_data = []
    val_acc_data = []
    training_acc_data = []

    print(f"--- Starting Supervised Training on {len(train_ds)} samples ---")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            if batch[0] is None:
                continue
            x, y = batch
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_correct += ((torch.sigmoid(pred) > 0.5).float() == y).sum().item()
            train_total += y.numel()  # Pixel-wise: count all pixels, not just batch items
                
        avg_train = train_loss / len(train_loader) if len(train_loader) > 0 else 0
        avg_train_acc = 100 * train_correct / (train_total + 1e-8)
        train_loss_data.append(avg_train)
        training_acc_data.append(avg_train_acc)
        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                if batch[0] is None:
                    continue
                x, y = batch
                x, y = x.to(DEVICE), y.to(DEVICE)            
                pred = model(x)
                val_loss += criterion(pred, y).item()

                val_correct += ((torch.sigmoid(pred) > 0.5).float() == y).sum().item()
                val_total += y.numel()
        
        avg_val = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_acc = 100 * val_correct / (val_total + 1e-8)
        val_loss_data.append(avg_val)
        val_acc_data.append(val_acc)

        print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train:.4f} | Train Acc: {avg_train_acc:.2f}% "
              f"| Val Loss: {avg_val:.4f} | Val Acc: {val_acc:.2f}%")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), SAVE_PATH)
            print("  >> Model saved.")
            trigger_times = 0
        else:
            trigger_times += 1
            print(f"  No improvement for {trigger_times} epoch(s).")
            if trigger_times >= patience:
                print("Early Stopping!")
                break

    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_data, label='Training Loss')
    plt.plot(val_loss_data, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("./plots/artifact_model_training_loss.png"); 
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(training_acc_data, label='Training Accuracy')
    plt.plot(val_acc_data, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig("./plots/artifact_model_training_accuracy.png")
    plt.close()
    print("Done.")


if __name__ == "__main__":
    train_supervised()