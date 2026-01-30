import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import os
from pathlib import Path

from model_architecture import FrequencyExpert

# CONFIG
DATA_FOLDER = "./data/processed_data" 
SAVE_PATH = "models/frequency_model.pth"
BATCH_SIZE = 32
LR = 0.001
EPOCHS = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FrequencyDataset(Dataset):
    def __init__(self, folder_path):
        self.files = list(Path(folder_path).rglob("*.pt"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            data = torch.load(self.files[idx])
            # Extract FFT Tensor [1, 256, 256]
            x = data['fft'].float()
            # Ensure correct shape
            if x.ndim == 2: x = x.unsqueeze(0)
            
            y = torch.tensor([data['label']], dtype=torch.float32)
            return x, y
        except Exception:
            return torch.zeros(1, 256, 256), torch.tensor([0.0])

def train():
    print(f"found device: {DEVICE}")
    
    # Setup Data
    dataset = FrequencyDataset(DATA_FOLDER)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    # Setup Model
    model = FrequencyExpert().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    # Loop
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
            
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                pred = (torch.sigmoid(model(X)) > 0.5).float()
                correct += (pred == y).sum().item()
                total += y.size(0)
        
        acc = 100 * correct / (total + 1e-8)
        print(f"Epoch {epoch+1} | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {acc:.2f}%")
        
        if acc >= best_acc:
            best_acc = acc
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), SAVE_PATH)

    print(f"Done. Saved to {SAVE_PATH}")

if __name__ == "__main__":
    train()