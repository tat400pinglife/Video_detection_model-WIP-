import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split
from pathlib import Path
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt

# Import Architecture
from model_architecture import PRNUBranch

# CONFIG
DATA_FOLDER = "./data/processed_data"
SAVE_PATH = "models/noise_model.pth"
BATCH_SIZE = 32
LR = 0.001
EPOCHS = 100 # adjust accordingly
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4

warnings.filterwarnings("ignore")


class RobustNoiseDataset(Dataset):
    def __init__(self, folder_path):
        all_files = list(Path(folder_path).rglob("*.pt"))
        print(f"Noise Dataset: Found {len(all_files)} total files, scanning labels...")

        self.files = []
        self.labels = []
        skipped = 0
        for f in all_files:
            parent = str(f.parent).lower()
            if "real" in parent:
                self.files.append(f)
                self.labels.append(0)
            elif "fake" in parent:
                self.files.append(f)
                self.labels.append(1)
            else:
                skipped += 1

        print(f"  Valid samples: {len(self.files)} | Skipped (unknown folder): {skipped}")

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        try:
            path = self.files[idx]
            data = torch.load(path, weights_only=False)

            if 'prnu' not in data:
                return None

            x = data['prnu'].float()  # [1, 256, 256]
            if x.ndim == 2:
                x = x.unsqueeze(0)

            y = torch.tensor([data['label']], dtype=torch.float32)
            return x, y

        except Exception as e:
            print(f"[WARN] Failed to load sample {idx}: {e}")
            return None

def drop_corrupt_collate(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0: return None, None
    inputs, labels = zip(*batch)
    return torch.stack(inputs), torch.stack(labels)

def train_noise_expert():
    print(f"Using device: {DEVICE}")
    
    # 1. Setup Data
    dataset = RobustNoiseDataset(DATA_FOLDER)
    if len(dataset) == 0: return

    # 2. Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    
    # 3. Balance Classes (Train Set Only)
    train_indices = train_set.indices
    train_labels = [dataset.labels[i] for i in train_indices]
    
    class_counts = np.bincount(train_labels)
    if len(class_counts) < 2:
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
    
    # 5. Initialize Model
    net = PRNUBranch().to(DEVICE)
    head = nn.Linear(32, 1).to(DEVICE)

    optimizer = optim.Adam(list(net.parameters()) + list(head.parameters()), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    
    best_acc = 0.0
    loss_data = []
    val_acc_data = []
    val_loss_data = []
    train_acc_data = []

    for epoch in range(EPOCHS):
        # TRAIN
        net.train(); head.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for X, y in train_loader:
            if X is None:
                continue
            X, y = X.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            feats = net(X)
            pred = head(feats)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += ((torch.sigmoid(pred) > 0.5).float() == y).sum().item()
            train_total += y.size(0)

        avg_train_loss = train_loss / max(len(train_loader), 1)
        avg_train_acc = 100 * train_correct / (train_total + 1e-8)
        loss_data.append(avg_train_loss)
        train_acc_data.append(avg_train_acc)

        # VALIDATE
        net.eval(); head.eval()
        correct = 0
        total = 0
        val_loss = 0

        with torch.no_grad():
            for X, y in val_loader:
                if X is None:
                    continue
                X, y = X.to(DEVICE), y.to(DEVICE)

                feats = net(X)
                logits = head(feats)  # Single forward pass
                val_loss += criterion(logits, y).item()

                correct += ((torch.sigmoid(logits) > 0.5).float() == y).sum().item()
                total += y.size(0)

        avg_val_loss = val_loss / max(len(val_loader), 1)
        acc = 100 * correct / (total + 1e-8)
        val_acc_data.append(acc)
        val_loss_data.append(avg_val_loss)

        print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.2f}% "
              f"| Val Loss: {avg_val_loss:.4f} | Val Acc: {acc:.2f}%")

        if acc > best_acc and acc > 50.0:
            best_acc = acc
            os.makedirs("models", exist_ok=True)
            full_state = {}
            for k, v in net.state_dict().items():
                full_state[f'net.{k}'] = v
            for k, v in head.state_dict().items():
                full_state[f'head.{k}'] = v

            torch.save(full_state, SAVE_PATH)
            print(f"  >> Saved Best Model ({acc:.2f}%)")


    print(f"Done. Best Val Acc: {best_acc:.2f}%")

    plt.figure(figsize=(10, 5))
    plt.plot(loss_data, label='Training Loss')
    plt.plot(val_loss_data, label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("noise_model_training_loss.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(train_acc_data, label='Training Accuracy')
    plt.plot(val_acc_data, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig("noise_model_training_accuracy.png")
    plt.close()


if __name__ == "__main__":
    train_noise_expert()