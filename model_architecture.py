import torch
import torch.nn as nn

# CLASSIFIER COMPONENTS

class SimpleBranch(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels, 16, 3, 1, 1),
            nn.BatchNorm2d(16), 
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Layer 2
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),  
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Layer 3
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten()
        )
    def forward(self, x): return self.net(x)
    
class PRNUBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.BatchNorm2d(8),  
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, 3, padding=1),
            nn.BatchNorm2d(16), 
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),  
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten()
        )
    def forward(self, x): return self.net(x)

class TinyDeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_branch = SimpleBranch(3)
        self.diff_branch = SimpleBranch(1)
        self.fft_branch = SimpleBranch(1)
        self.prnu_branch = PRNUBranch()
        
        self.feature_size = 64 * 32 * 32 
        self.prnu_size = 32 * 32 * 32 
        
        self.prnu_gate = nn.Sequential(nn.Linear(self.prnu_size, 1), nn.Sigmoid())
        
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size * 3 + self.prnu_size * 2, 128), 
            nn.BatchNorm1d(128), 
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
    
# ARTIFACT HUNTER (U-NET)
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

# TEMPORAL DETECTOR (LSTM)
class TemporalDetector(nn.Module):
    def __init__(self, pretrained_cnn=None):
        super().__init__()
        # 1. CNN Feature Extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten()
        )
        
        # 2. If we provided a pretrained classifier, COPY its weights
        if pretrained_cnn is not None:
            # We copy the layers from the pretrained SimpleBranch
            # pretrained_cnn is the 'rgb_branch' of TinyDeepfakeDetector
            print(">> Loading pretrained CNN weights into Temporal Model...")
            self.cnn.load_state_dict(pretrained_cnn.net.state_dict())
            
            # OPTIONAL: Freeze the CNN so we ONLY train the LSTM?
            # for param in self.cnn.parameters():
            #     param.requires_grad = False
        
        # 3. LSTM (Input size depends on CNN output)
        # 64 channels * 32 * 32 spatial size = 65536 features
        self.lstm = nn.LSTM(
            input_size=64 * 32 * 32, 
            hidden_size=128, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=True
        )
        
        # 4. Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 * 2, 64), # *2 for bidirectional
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        b, t, c, h, w = x.size()
        # Fold time into batch
        c_in = x.view(b * t, c, h, w)
        # Extract features
        features = self.cnn(c_in)
        # Unfold time
        features_seq = features.view(b, t, -1)
        # Scan time
        lstm_out, _ = self.lstm(features_seq)
        return self.classifier(lstm_out)