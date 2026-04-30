import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import os
from pathlib import Path
import random
import matplotlib.pyplot as plt

# Import your model
from model_architecture import TemporalDetector
import warnings
warnings.filterwarnings("ignore")

# CONFIG
DATA_FOLDER = "./data/processed_data"
SAVE_PATH = "models/temporal_lstm.pth"
BATCH_SIZE = 16 # unless you have a giga device dont increase this by too much
SEQ_LEN = 25     
LR = 0.00001
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE = 8 # Early stopping patience


class TemporalSequenceDataset(Dataset):
    def __init__(self, folder_path):
        self.files = list(Path(folder_path).rglob("*.pt"))
        print(f"Temporal Dataset: Found {len(self.files)} samples.")
        self.weights = torch.tensor([0.299, 0.587, 0.114]).view(3, 1, 1)

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        try:
            path = self.files[idx]
            data = torch.load(path, weights_only=False)

            rgb = data['rgb_batch'].float() / 255.0  # [32, 3, 256, 256]

            gray = (rgb * self.weights.to(rgb.device)).sum(dim=1)  # [32, 256, 256]

            diff_seq = torch.abs(gray[1:] - gray[:-1])  # [31, 256, 256]
            diff_seq = diff_seq.unsqueeze(1)             # [31, 1, 256, 256]

            # Slice SEQ_LEN so dataset output always matches model expectation.
            diff_seq = diff_seq[:SEQ_LEN]  # [SEQ_LEN, 1, 256, 256]

            y = torch.tensor([data['label']], dtype=torch.float32)
            return diff_seq, y

        except Exception as e:
            print(f"Failed to load sample {idx}: {e}")
            return None

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def train():
    print(f"Using Device: {DEVICE}")
    
    train_loss_data = []
    val_loss_data = []
    val_acc_data = []
    train_acc_data = []

    # 1. Setup Data
    full_dataset = TemporalSequenceDataset(DATA_FOLDER)

    train_size = int(0.8 * len(full_dataset))
    val_size   = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,num_workers=4, collate_fn=collate_fn)
    val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,num_workers=4, collate_fn=collate_fn)

    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # 2. Setup Model & Upgraded Optimizer
    model = TemporalDetector(sequence_length=SEQ_LEN).to(DEVICE)
    # change to adamw since it was overfitting
    # 1e-3 is a strong starting penalty to prevent weight memorization.
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    # 3. Training Loop Variables
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss    = 0
        train_correct = 0
        train_total   = 0

        for batch in train_loader:
            if batch is None:
                continue

            X, y = batch
            X, y = X.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item()
            train_correct += ((torch.sigmoid(pred) > 0.5).float() == y).sum().item()
            train_total   += y.size(0)

        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
        avg_train_acc  = 100 * train_correct / (train_total + 1e-8)
        train_loss_data.append(avg_train_loss)
        train_acc_data.append(avg_train_acc)

        # --- VALIDATION ---
        model.eval()
        correct  = 0
        total    = 0
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue

                diff_seq, labels = [x.to(DEVICE) for x in batch]
                output = model(diff_seq)

                val_loss += criterion(output, labels).item()

                predicted_labels = (torch.sigmoid(output) > 0.5).float()
                correct += (predicted_labels == labels).sum().item()
                total   += labels.size(0)

        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        acc          = 100 * correct / (total + 1e-8)
        val_acc_data.append(acc)
        val_loss_data.append(avg_val_loss)

        print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.2f}% "
              f"| Val Loss: {avg_val_loss:.4f} | Val Acc: {acc:.2f}%")

        # Checkpoint based on LOWEST Validation Loss, not highest accuracy
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  >> Best model saved — Val Loss: {best_val_loss:.4f} (Acc: {acc:.2f}%)")
        else:
            epochs_no_improve += 1
            print(f"  -- No improvement in Val Loss for {epochs_no_improve} epoch(s).")
            
            if epochs_no_improve >= PATIENCE:
                print(f"\nValidation loss has not improved for {PATIENCE} epochs.")
                break

    print(f"\nTraining complete. Best Val Loss: {best_val_loss:.4f}")

    # Minimalist Plots
    Path("./plots").mkdir(exist_ok=True)
    
    # Loss Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_loss_data, label="Train Loss", color='#1f77b4', linewidth=2)
    ax.plot(val_loss_data, label="Val Loss", color='#ff7f0e', linewidth=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel("Epoch", labelpad=10)
    ax.set_ylabel("Loss", labelpad=10)
    ax.legend(frameon=False)
    plt.title("Training and Validation Loss", pad=15)
    plt.tight_layout()
    plt.savefig("./plots/temporal_model_training_loss.png", dpi=300)
    plt.close()

    # Accuracy Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_acc_data, label="Train Accuracy", color='#2ca02c', linewidth=2)
    ax.plot(val_acc_data, label="Val Accuracy", color='#d62728', linewidth=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel("Epoch", labelpad=10)
    ax.set_ylabel("Accuracy (%)", labelpad=10)
    ax.legend(frameon=False)
    plt.title("Training and Validation Accuracy", pad=15)
    plt.tight_layout()
    plt.savefig("./plots/temporal_model_training_accuracy.png", dpi=300)
    plt.close()

    print("Done.")

if __name__ == "__main__":
    train()