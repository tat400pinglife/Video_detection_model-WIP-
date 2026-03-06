import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# Import the architecture
from model_architecture import MoE_Investigator

# CONFIGURATION
DATA_FOLDER = "./data/processed_data"
BATCH_SIZE = 16    
LR = 0.001        
EPOCHS = 50 # adjust accordingly
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

            def fix(t):
                if t.ndim == 2: t = t.unsqueeze(0)
                if t.ndim == 4: t = t.squeeze(0)
                return torch.nan_to_num(t.float(), nan=0.0)

            batch_frames = data['rgb_batch'].float() / 255.0  # [N, 3, 256, 256]
            mid_idx      = batch_frames.shape[0] // 2
            rgb          = batch_frames[mid_idx]               # [3, 256, 256]
            gray_weights = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            gray         = (batch_frames * gray_weights).sum(dim=1)  # [N, 256, 256]
            diff_seq     = torch.abs(gray[1:] - gray[:-1]).unsqueeze(1)  # [N-1, 1, 256, 256]
            prnu  = fix(data['prnu'])
            fft   = fix(data['fft'])
            audio = fix(data['audio'])
            label = torch.tensor([data['label']], dtype=torch.float32)

            # Slice diff_seq to seq_len using a centre window.
            if diff_seq.shape[0] < self.seq_len:
                pad      = torch.zeros(self.seq_len - diff_seq.shape[0], 1, 256, 256)
                diff_seq = torch.cat([diff_seq, pad], dim=0)

            mid   = diff_seq.shape[0] // 2
            start = max(0, mid - (self.seq_len // 2))
            if start + self.seq_len > diff_seq.shape[0]:
                start = 0
            diff_seq = diff_seq[start : start + self.seq_len]  # [seq_len, 1, 256, 256]

            return rgb, diff_seq, prnu, fft, audio, label

        except Exception as e:
            print(f"[WARN] Failed to load sample {idx}: {e}")
            return None

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0: return None
    return torch.utils.data.dataloader.default_collate(batch)


def train():
    print(f"Initializing MoE Training on {DEVICE}...")
    train_loss_data = []
    val_loss_data = []
    val_acc_data = []
    train_acc_data = []


    full_dataset = ForensicDataset(DATA_FOLDER, seq_len=SEQ_LEN)
    if len(full_dataset) == 0: return

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Initialize MoE with the NEW LSTM path
    model = MoE_Investigator(
        temp_path="models/temporal_lstm.pth",
        art_path="models/artifact_model.pth",
        noise_path="models/noise_model.pth",
        freq_path="models/frequency_model.pth",
        audio_path="models/audio_model.pth"
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)

    # MoE forward() outputs probabilities (already sigmoid'd via expert fusion),
    criterion = nn.BCELoss()
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        avg_weights = np.zeros(5)
        train_correct = 0
        train_total = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch in loop:
            if batch is None:
                continue

            rgb, diff_seq, prnu, fft, audio, labels = [x.to(DEVICE) for x in batch]

            optimizer.zero_grad()

            # Model output is already a probability (0-1) from the MoE fusion sigmoid
            predictions, route_weights = model(rgb, diff_seq, prnu, fft, audio)
            loss = criterion(predictions, labels)
            
            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            avg_weights += route_weights.detach().cpu().numpy().mean(axis=0)
            train_correct += ((predictions > 0.5).float() == labels).sum().item()
            train_total += labels.size(0)

            loop.set_postfix(loss=f"{loss.item():.4f}")

        num_batches = max(len(train_loader), 1)
        avg_train_loss = running_loss / num_batches
        avg_train_acc = 100 * train_correct / (train_total + 1e-8)
        avg_weights /= num_batches

        train_loss_data.append(avg_train_loss)
        train_acc_data.append(avg_train_acc)

        val_acc, avg_val_loss = evaluate(model, val_loader, criterion)
        model.train()  # Explicitly restore train mode after evaluate()

        val_loss_data.append(avg_val_loss)
        val_acc_data.append(val_acc)

        w_str = " | ".join([f"{x*100:.1f}%" for x in avg_weights])
        print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.2f}% "
              f"| Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"  Router Weights [ Motn | Artf | Nois | Audi | Freq ]")
        print(f"                 [ {w_str} ]")

        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/router_weights.pth")
            print(f"  >> Best model saved — Val Acc: {best_acc:.2f}%")

    print(f"\nTraining complete. Best Val Acc: {best_acc:.2f}%")

    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_data, label='Training Loss')
    plt.plot(val_loss_data, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('moe_model_training_loss.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(train_acc_data, label='Training Accuracy')
    plt.plot(val_acc_data, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.savefig('moe_model_training_accuracy.png')
    plt.close()

    print("Done.")


def evaluate(model, loader, criterion):
    """Returns (accuracy %, avg_val_loss) over the full loader."""
    model.eval()
    correct = 0
    total = 0
    val_loss = 0

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            rgb, diff_seq, prnu, fft, audio, labels = [x.to(DEVICE) for x in batch]
            preds, _ = model(rgb, diff_seq, prnu, fft, audio)

            val_loss += criterion(preds, labels).item()
            correct += ((preds > 0.5).float() == labels).sum().item()
            total += labels.size(0)

    avg_val_loss = val_loss / max(len(loader), 1)
    accuracy = 100 * correct / (total + 1e-8)
    return accuracy, avg_val_loss


if __name__ == "__main__":
    train()