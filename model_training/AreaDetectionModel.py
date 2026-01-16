import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import random
from pathlib import Path

from model_architecture import ArtifactSegmentor

def create_glitch_batch(real_imgs_batch):
    """
    Takes a batch of REAL images (B, H, W, 3) and creates FAKES + MASKS.
    Applies aggressive artifacts (Pixelation + Color Shift) to lower loss.
    """
    # Ensure inputs are 0.0 - 1.0
    if real_imgs_batch.max() > 1.0:
        real_imgs_batch = real_imgs_batch / 255.0
        
    batch_size, h, w, c = real_imgs_batch.shape
    inputs, masks = [], []
    
    for i in range(batch_size):
        img = real_imgs_batch[i] # (H, W, 3) Float
        
        # A. Create Mask
        mask = np.zeros((h, w), dtype=np.float32)
        cx, cy = random.randint(50, 200), random.randint(50, 200)
        
        # Random Shape: Circle or Square
        if random.random() > 0.5:
            radius = random.randint(30, 80)
            cv2.circle(mask, (cx, cy), radius, 1.0, -1)
        else:
            size = random.randint(40, 100)
            x1, y1 = max(0, cx - size//2), max(0, cy - size//2)
            cv2.rectangle(mask, (x1, y1), (x1+size, y1+size), 1.0, -1)
            
        # Soften edges slightly
        mask_blur = cv2.GaussianBlur(mask, (15, 15), 0)[:,:,None] # change paramters for more/less blur, lower parameters looks like a cutout
        
        # B. Create Nasty Artifact
        # 1. Pixelation (Downscale -> Upscale)
        scale = random.uniform(0.1, 0.3) # 0.1 to 0.3 of original size
        small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        artifact = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # 2. Color Shift (Simulate bad lighting matching)
        # Randomly boost/suppress Green or Red channels
        c_idx = random.randint(0, 2)
        artifact[:, :, c_idx] *= random.uniform(0.8, 1.2) # range 0.8 to 1.2 20% brightness/color change
        artifact = np.clip(artifact, 0, 1)

        # --- C. Blend ---
        # Real * (1-Mask) + Fake * Mask
        blended = img * (1 - mask_blur) + artifact * mask_blur
        
        # Add Noise (Anti-overfitting)
        noise = np.random.normal(0, 0.005, blended.shape).astype(np.float32)
        blended = np.clip(blended + noise, 0, 1)
        
        inputs.append(blended)
        masks.append(mask) # Train on the sharp mask
        
    # Convert to PyTorch Tensors
    # Inputs: (B, H, W, 3) -> (B, 3, H, W)
    t_inputs = torch.tensor(np.array(inputs)).permute(0,3,1,2).float()
    # Masks: (B, H, W) -> (B, 1, H, W)
    t_masks = torch.tensor(np.array(masks)).unsqueeze(1).float()
    
    return t_inputs, t_masks

class RealTensorDataset(Dataset):
    def __init__(self, root_dir):
        self.files = list(Path(root_dir).rglob("*.pt"))
        
        if len(self.files) == 0:
            print(f"ERROR: No .pt files found in {root_dir}")
            raise RuntimeError("Dataset empty?")
            
        print(f">> Found {len(self.files)} tensor files for training.")
        
    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        try:
            # Load tensor file
            data = torch.load(self.files[idx], weights_only=False)
            
            # Extract RGB frames: (32, 3, 256, 256)
            frames_tensor = data['rgb']
            
            # Pick ONE random frame from this video to train on
            random_idx = random.randint(0, frames_tensor.shape[0] - 1)
            frame = frames_tensor[random_idx] # (3, 256, 256)
            
            # Glitch Generator expects (H, W, 3) for OpenCV
            # Permute: (3, H, W) -> (H, W, 3)
            frame = frame.permute(1, 2, 0)
            
            return frame # Float Tensor 0.0-1.0
            
        except Exception as e:
            # Return empty black frame on error
            return torch.zeros((256, 256, 3), dtype=torch.float32)

def train_artifact_expert():
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training Artifact Expert on {device} ---")
    
    # 1. Load Data (Tensors)
    # Point this to your PROCESSED REAL data
    dataset = RealTensorDataset("./data/processed_data/real")
    
    # change batch size when doing the real dataset
    loader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)
    
    # 2. Initialize Model
    model = ArtifactSegmentor().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.BCEWithLogitsLoss()
    
    print("Starting Training...")
    
    epochs = 60
    for epoch in range(1, epochs+1):
        total_loss = 0
        
        model.train()
        for i, real_batch in enumerate(loader):
            # A. Create Fakes on the Fly
            # real_batch is (B, 256, 256, 3) on CPU
            inputs, masks = create_glitch_batch(real_batch.numpy())
            
            # B. Move to Device
            inputs = inputs.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            
            # C. Forward & Backward
            logits = model(inputs) # (B, 1, 256, 256)
            loss = criterion(logits, masks)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")
        if avg_loss < 0.15:
            print(">> Early Stopping: Model is performing well.")
            break
        
        # Save Debug Image (Every 5 Epochs)
        if epoch % 5 == 0:
            with torch.no_grad():
                # Grab first item in batch
                pred = torch.sigmoid(logits[0]).cpu().numpy().squeeze()
                in_img = inputs[0].permute(1,2,0).cpu().numpy() # Back to HWC
                gt_mask = masks[0].cpu().numpy().squeeze()
                
                # Stack images side-by-side
                debug_img = np.hstack([in_img, np.stack([pred]*3, axis=-1), np.stack([gt_mask]*3, axis=-1)])
                debug_img = (debug_img * 255).astype(np.uint8)
                cv2.imwrite(f"debug_epoch_{epoch}.png", cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))

    # 3. Save Model
    torch.save(model.state_dict(), "./models/unet_artifact_hunter.pth")
    print("\n>> Training Complete.")
    print(">> Saved 'unet_artifact_hunter.pth'")

if __name__ == "__main__":
    train_artifact_expert()