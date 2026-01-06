import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. THE EXPERTS
# ==========================================

class PRNUBranch(nn.Module):
    def __init__(self):
        super().__init__()
        # Added Dropout as discussed to prevent 0.001 loss (Overfitting)
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
            nn.Dropout(0.3),
            nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Dropout(0.4),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten()
        )
    def forward(self, x): return self.net(x)

class ArtifactSegmentor(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = self.conv_block(3, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self.conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self.conv_block(64, 128)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = self.conv_block(128 + 64, 64)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = self.conv_block(64 + 32, 32)
        self.final = nn.Conv2d(32, 1, kernel_size=1)

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        d2 = self.up2(e3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return self.final(d1)

class TemporalDetector(nn.Module):
    def __init__(self, pretrained_cnn=None):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten()
        )
        if pretrained_cnn:
             self.cnn.load_state_dict(pretrained_cnn.state_dict())
        self.lstm = nn.LSTM(input_size=64 * 32 * 32, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(nn.Linear(128 * 2, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        b, t, c, h, w = x.size()
        c_in = x.view(b * t, c, h, w)
        features = self.cnn(c_in)
        features_seq = features.view(b, t, -1)
        lstm_out, _ = self.lstm(features_seq)
        return self.classifier(lstm_out)

# ==========================================
# 2. THE ROUTER
# ==========================================

class InvestigatorRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(4),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(4),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten()
        )
        self.fc = nn.Linear(64, 3)

    def forward(self, x):
        features = self.cnn(x)
        logits = self.fc(features)
        return F.softmax(logits, dim=1)

# ==========================================
# 3. THE MOE SYSTEM
# ==========================================

class MoE_Investigator(nn.Module):
    def __init__(self, temp_path=None, art_path=None, noise_path=None):
        super().__init__()
        
        self.router = InvestigatorRouter()
        
        self.expert_temp = TemporalDetector()
        self.expert_art  = ArtifactSegmentor()
        self.expert_noise_net = PRNUBranch()
        self.expert_noise_head = nn.Linear(32*32*32, 1)

        # LOAD WEIGHTS
        if temp_path: self._load_safe(self.expert_temp, temp_path, "Temporal")
        if art_path:  self._load_safe(self.expert_art, art_path, "Artifact")
        if noise_path: self._load_noise_smart(noise_path)
            
        # FREEZE EXPERTS
        self._freeze(self.expert_temp)
        self._freeze(self.expert_art)
        self._freeze(self.expert_noise_net)

    def forward(self, rgb_mid, rgb_seq, prnu_var):
        weights = self.router(rgb_mid) # (B, 3)
        
        out_temp = torch.sigmoid(self.expert_temp(rgb_seq)).mean(dim=1)
        
        out_art_map = torch.sigmoid(self.expert_art(rgb_mid))
        out_art = out_art_map.flatten(1).max(1)[0].unsqueeze(1)
        
        out_noise = torch.sigmoid(self.expert_noise_head(self.expert_noise_net(prnu_var)))
        
        experts = torch.cat([out_temp, out_art, out_noise], dim=1)
        verdict = (experts * weights).sum(dim=1, keepdim=True)
        
        return verdict, weights

    def _freeze(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def _load_safe(self, model, path, name):
        try:
            model.load_state_dict(torch.load(path, weights_only=True))
            print(f">> Loaded {name} Expert.")
        except Exception as e:
            print(f"!! Failed to load {name} Expert: {e}")

    # --- FIX: SMART LOADING LOGIC ---
    def _load_noise_smart(self, path):
        try:
            state = torch.load(path, weights_only=True)
            
            # Case 1: File is from train_noise.py (Standalone Expert)
            # Keys look like: 'net.0.weight' or '0.weight'
            if any(k.startswith('net.') for k in state.keys()):
                self.expert_noise_net.load_state_dict(state, strict=False)
                print(">> Loaded Noise Expert (Standalone Format).")
                
            # Case 2: File is from old Classifier (Full Model)
            # Keys look like: 'prnu_branch.net.0.weight'
            elif any('prnu_branch' in k for k in state.keys()):
                prnu_state = {k.replace('prnu_branch.net.', 'net.'): v 
                              for k, v in state.items() if 'prnu_branch' in k}
                self.expert_noise_net.load_state_dict(prnu_state)
                print(">> Loaded Noise Expert (Extracted from Classifier).")
                
            else:
                print("!! Warning: Noise Expert file format unrecognized (keys mismatch).")

        except Exception as e:
            print(f"!! Failed to load Noise Expert: {e}")