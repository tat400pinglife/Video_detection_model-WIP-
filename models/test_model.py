import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random

# --- 1. DATASET LOADER ---
class DeepfakeDataset(Dataset):
    def __init__(self, root_dir):
        # Find all .pt files
        self.files = list(Path(root_dir).rglob("*.pt"))
        if not self.files:
            print(f"Warning: No .pt files found in {root_dir}")
        random.shuffle(self.files) 
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        
        # FIX: Explicitly set weights_only=False to fix the FutureWarning/Security warning
        data = torch.load(path, weights_only=False)
        
        # We extracted 32 frames. 
        # We pick the MIDDLE frame (Index 16) for this PoC.
        mid_idx = 16 
        
        rgb = data['rgb'][mid_idx]    # Shape: (3, 256, 256)
        diff = data['diff'][mid_idx]  # Shape: (1, 256, 256)
        fft = data['fft'][mid_idx]    # Shape: (1, 256, 256)
        # PRNU variance across time (T,1,H,W) → (1,H,W)
        prnu_var = data['prnu'].var(dim=0)
        # Optional: also keep averaged PRNU fingerprint
        prnu_mean = data['prnu'].mean(dim=0)



        
        label = torch.tensor(data['label'], dtype=torch.float32)
        return rgb, diff, fft, prnu_mean, prnu_var, label

# --- 2. THE MODEL ARCHITECTURE ---

class SimpleBranch(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            # --- Layer 1 (general neighborhood noise) ---
            # Input: 256x256
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # Reduces size by 2 -> 128x128
            
            # --- Layer 2 (frame texture) ---
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # Reduces size by 2 -> 64x64

            # --- Layer 3 (regional features/consistency) ---
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # Reduces size by 2 -> 32x32

            #  --- Flatten ---
            nn.Flatten() 
        )
        
    def forward(self, x):
        return self.net(x)
    
class PRNUBranch(nn.Module):
    #prnu as a classifier becomes meaningless when dealing with larger quantities of signals
    # this is lowered with a gate such that it is only used when necessary
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten()
        )

    def forward(self, x):
        return self.net(x)

class TinyDeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.rgb_branch = SimpleBranch(in_channels=3)
        self.diff_branch = SimpleBranch(in_channels=1)
        self.fft_branch = SimpleBranch(in_channels=1)
        self.prnu_branch = PRNUBranch()
        
        # MATH: 
        # Final image size is 32x32. 
        # Final depth is 64 channels.
        # 64 * 32 * 32 = 65,536 features per branch.
        self.feature_size = 64 * 32 * 32 
        self.prnu_size = 32 * 32 * 32 
        
        self.prnu_gate = nn.Sequential(
            nn.Linear(self.prnu_size, 1),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            # 65,536 * 4 branches = 262,144 inputs
            nn.Linear(self.feature_size * 3 + self.prnu_size * 2, 128), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1) 
        )

    def forward(self, rgb, diff, fft, prnu_mean, prnu_var):
        f_rgb = self.rgb_branch(rgb)
        f_diff = self.diff_branch(diff)
        f_fft = self.fft_branch(fft)

        f_prnu_mean = self.prnu_branch(prnu_mean)
        f_prnu_var = self.prnu_branch(prnu_var)

        gate = self.prnu_gate(f_prnu_mean)
        f_prnu_mean = f_prnu_mean * gate

        combined = torch.cat([f_rgb, f_diff, f_fft, f_prnu_mean, f_prnu_var], dim=1)
        return self.classifier(combined)

# --- 3. TRAINING LOOP ---
def train_one_epoch():
    dataset = DeepfakeDataset("./data/frames") 
    
    if len(dataset) == 0:
        print("Error: No .pt files found. Did you run process_data.py?")
        return

    # Drop_last=True prevents the crash if the last batch is size 1
    # Alternatively, use .view(-1) as I implemented below
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    model = TinyDeepfakeDetector()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss() 

    print(f"Training on {len(dataset)} items (Resolution: 256x256)...")
    
    for epoch in range(1, 10):
        total_loss = 0
        correct = 0
        total = 0
        
        model.train()
        for rgb, diff, fft, prnu_mean, prnu_var,labels in loader:
            optimizer.zero_grad()

            outputs = model(rgb, diff, fft, prnu_mean, prnu_var).view(-1)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += len(labels)
            
        print(f"Epoch {epoch} | Loss: {total_loss/len(loader):.4f} | Acc: {correct/total:.1%}")

    torch.save(model.state_dict(), "poc_model_256.pth")
    print("\nTraining Complete! Model saved as 'poc_model_256.pth'")

if __name__ == "__main__":
    train_one_epoch()