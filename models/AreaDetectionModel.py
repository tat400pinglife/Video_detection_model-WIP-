import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import cv2
import random
from model_architecture import ArtifactSegmentor

class RealFramesDataset(Dataset):
    def __init__(self, root_dir):
        # We only want REAL frames to synthetically damage them
        self.files = list((Path(root_dir) / "real").rglob("*.pt"))
        if not self.files: print("Warning: No real frames found for U-Net training.")
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        try:
            data = torch.load(self.files[idx], weights_only=False)
            # Pick a random frame from the stack
            rgb = data['rgb'][random.randint(0, 15)].permute(1, 2, 0).numpy()
            return rgb
        except: return np.zeros((256, 256, 3), dtype=np.float32)
        
def create_glitch_batch(real_imgs):
    """
    Simulates Ghosting Artifacts by blending two different frames.
    (This was the previous version of the logic).
    """
    batch_size, h, w, c = real_imgs.shape
    
    # Shift the images so we blend Image A with Image B (creating a ghost)
    imgs_B = np.roll(real_imgs, 1, axis=0)
    
    inputs, masks = [], []
    for i in range(batch_size):
        # 1. Create a random mask
        mask = np.zeros((h, w), dtype=np.float32)
        cx, cy = random.randint(50, 200), random.randint(50, 200)
        radius = random.randint(30, 60)
        
        cv2.circle(mask, (cx, cy), radius, 1.0, -1)
        
        # Soften the edges so it isn't a perfect cutout
        mask = cv2.GaussianBlur(mask, (15, 15), 0)[:,:,None]
        
        # 2. Blend the two images
        # Where mask is 1, use Image B. Where mask is 0, use Image A.
        blended = mask * imgs_B[i] + (1 - mask) * real_imgs[i]
        
        inputs.append(blended)
        masks.append(mask)
        
    return torch.tensor(np.array(inputs)).permute(0,3,1,2).float(), torch.tensor(np.array(masks)).permute(0,3,1,2).float()

def train_artifact_expert():
    device = torch.device("cpu")
    dataset = RealFramesDataset("./data/frames")
    loader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)
    
    model = ArtifactSegmentor().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    print("Training Artifact Expert...")
    for epoch in range(1, 21):
        total_loss = 0
        for real_batch in loader:
            inputs, targets = create_glitch_batch(real_batch.numpy())
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch} | Loss: {total_loss/len(loader):.4f}")
        
    torch.save(model.state_dict(), "unet_artifact_hunter.pth")
    print("Artifact Expert Saved.")

if __name__ == "__main__":
    train_artifact_expert()