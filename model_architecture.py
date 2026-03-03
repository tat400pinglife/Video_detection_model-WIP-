import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. EXPERT MODELS

class AudioExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)
        )
    def forward(self, x): return self.net(x)

class PRNUBranch(nn.Module): # Noise
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.3),
            nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.4),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten()
        )
    def forward(self, x): return self.net(x)

class FrequencyExpert(nn.Module):
    def __init__(self):
        super().__init__()
        # Takes 1-channel FFT Spectrum [B, 1, 256, 256]
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 256), nn.ReLU(), nn.Linear(256, 1)
        )
    def forward(self, x): return self.net(x)

class ArtifactSegmentor(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = self.conv_block(3, 32); self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self.conv_block(32, 64); self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self.conv_block(64, 128)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = self.conv_block(128 + 64, 64)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = self.conv_block(64 + 32, 32)
        self.final = nn.Conv2d(32, 1, kernel_size=1)
    def conv_block(self, in_c, out_c):
        return nn.Sequential(nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(), nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU())
    def forward(self, x):
        e1 = self.enc1(x); p1 = self.pool1(e1); e2 = self.enc2(p1); p2 = self.pool2(e2); e3 = self.enc3(p2)
        d2 = self.up2(e3); d2 = torch.cat([d2, e2], dim=1); d2 = self.dec2(d2)
        d1 = self.up1(d2); d1 = torch.cat([d1, e1], dim=1); d1 = self.dec1(d1)
        return self.final(d1)

class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        # x shape: [B, S, C, H, W]
        b, s, c, h, w = x.size()
        # Merge Batch and Sequence for the CNN: [B*S, C, H, W]
        x_reshaped = x.view(b * s, c, h, w)
        # Pass through CNN
        y = self.module(x_reshaped)
        # Reshape back to sequence: [B, S, Features]
        return y.view(b, s, -1)

class TemporalDetector(nn.Module):
    def __init__(self, sequence_length=5):
        super().__init__()
        
        # 1. The Eye (Spatial CNN)
        # We strip the final layers to get raw features
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten()
        )
        # Wrap it
        self.time_distributed = TimeDistributed(self.cnn_encoder)
        # Feature size calculation: 256 -> 128 -> 64 -> 32
        # Final shape: 64 channels * 32 * 32
        self.feature_size = 64 * 32 * 32 
        # 2. The Memory (LSTM)
        self.lstm = nn.LSTM(
            input_size=self.feature_size,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        # 3. The Verdict
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        # Input: [Batch, Seq, 1, 256, 256]
        # 1. Get features for every frame
        # Out: [Batch, Seq, 65536]
        seq_features = self.time_distributed(x)
        
        # 2. Process time
        # Out: [Batch, Seq, 256]
        lstm_out, _ = self.lstm(seq_features)
        last_step = lstm_out[:, -1, :]
        
        return self.fc(last_step)
    
# class TemporalDetector(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Input is now 1-channel Diff Tensor [B, 1, 256, 256], not a sequence
#         self.cnn = nn.Sequential(
#             nn.Conv2d(1, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
#             nn.Conv2d(16, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
#             nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
#             nn.Flatten(),
#             nn.Linear(64 * 32 * 32, 128), nn.ReLU(), nn.Linear(128, 1)
#         )
#     def forward(self, x):
#         return self.cnn(x)

# 2. ROUTER & SYSTEM

class InvestigatorRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(4),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(4),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten()
        )
        self.fc = nn.Linear(64, 5)

    def forward(self, x):
        features = self.cnn(x)
        logits = self.fc(features)
        return F.softmax(logits, dim=1)

# class MoE_Investigator(nn.Module):
#     def __init__(self, temp_path=None, art_path=None, noise_path=None, audio_path=None, freq_path=None):
#         super().__init__()
        
#         self.router = InvestigatorRouter()
        
#         self.expert_temp  = TemporalDetector()  
#         self.expert_art   = ArtifactSegmentor()
#         self.expert_audio = AudioExpert() 
#         self.expert_freq  = FrequencyExpert()  
#         self.expert_noise_net  = PRNUBranch()
#         self.expert_noise_head = nn.Linear(32*32*32, 1)

#         # LOAD WEIGHTS
#         if temp_path: self._load_safe(self.expert_temp, temp_path, "Temporal(Diff)")
#         if art_path:  self._load_safe(self.expert_art, art_path, "Artifact")
#         if audio_path: self._load_safe(self.expert_audio, audio_path, "Audio")
#         if freq_path: self._load_safe(self.expert_freq, freq_path, "Frequency")
#         if noise_path: self._load_noise_smart(noise_path)
            
#         # FREEZE EVERYTHING
#         self._freeze(self.expert_temp)
#         self._freeze(self.expert_art)
#         self._freeze(self.expert_audio)
#         self._freeze(self.expert_freq)
#         self._freeze(self.expert_noise_net)

#     def forward(self, rgb_mid, diff_tensor, prnu_var, fft_var, audio_spec):
#         # Inputs:
#         # - rgb_mid: (B, 3, 256, 256) 
#         # - diff_tensor: (B, 1, 256, 256)
#         # - prnu_var: (B, 1, 256, 256) 
#         # - fft_var: (B, 1, 256, 256)
#         # - audio_spec: (B, 1, 128, 128) 

#         weights = self.router(rgb_mid) # (B, 5)
        
#         # 1. Temporal (Diff) Expert
#         out_temp = torch.sigmoid(self.expert_temp(diff_tensor))
        
#         # 2. Artifact Expert
#         out_art_map = torch.sigmoid(self.expert_art(rgb_mid))
#         out_art = out_art_map.flatten(1).max(1)[0].unsqueeze(1)
        
#         # 3. Noise (PRNU) Expert
#         out_noise = torch.sigmoid(self.expert_noise_head(self.expert_noise_net(prnu_var)))

#         # 4. Frequency (FFT) Expert
#         out_freq = torch.sigmoid(self.expert_freq(fft_var))
        
#         # 5. Audio Expert (Handle missing audio)
#         if audio_spec is not None and audio_spec.sum() != 0:
#             out_audio = torch.sigmoid(self.expert_audio(audio_spec))
#         else:
#             out_audio = torch.full_like(out_noise, 0.5)

#         # Fusion
#         experts = torch.cat([out_temp, out_art, out_noise, out_audio, out_freq], dim=1)
#         verdict = (experts * weights).sum(dim=1, keepdim=True)
        
#         return verdict, weights

#     def _freeze(self, module):
#         for param in module.parameters(): param.requires_grad = False
#     def _load_safe(self, model, path, name):
#         try: model.load_state_dict(torch.load(path, weights_only=True)); print(f">> Loaded {name} Expert.")
#         except Exception as e: print(f"!! Failed to load {name} Expert: {e}")
#     def _load_noise_smart(self, path):
#         try:
#             state = torch.load(path, weights_only=True)
#             if any(k.startswith('net.') for k in state.keys()): self.expert_noise_net.load_state_dict(state, strict=False)
#             elif any('prnu_branch' in k for k in state.keys()):
#                 prnu_state = {k.replace('prnu_branch.net.', 'net.'): v for k, v in state.items() if 'prnu_branch' in k}
#                 self.expert_noise_net.load_state_dict(prnu_state)
#             print(">> Loaded Noise Expert.")
#         except: pass

class MoE_Investigator(nn.Module):
    def __init__(self, temp_path=None, art_path=None, noise_path=None, audio_path=None, freq_path=None):
        super().__init__()
        
        self.router = InvestigatorRouter()
        
        # EXPERTS
        self.expert_temp  = TemporalDetector(sequence_length=5)   
        self.expert_art   = ArtifactSegmentor()
        self.expert_audio = AudioExpert() 
        self.expert_freq  = FrequencyExpert()   
        self.expert_noise_net  = PRNUBranch()
        self.expert_noise_head = nn.Linear(32*32*32, 1)

        # LOAD WEIGHTS
        if temp_path: self._load_safe(self.expert_temp, temp_path, "Temporal(LSTM)")
        if art_path:  self._load_safe(self.expert_art, art_path, "Artifact")
        if audio_path: self._load_safe(self.expert_audio, audio_path, "Audio")
        if freq_path: self._load_safe(self.expert_freq, freq_path, "Frequency")
        if noise_path: self._load_noise_smart(noise_path)
            
        # FREEZE EVERYTHING (Train only Router)
        self._freeze(self.expert_temp)
        self._freeze(self.expert_art)
        self._freeze(self.expert_audio)
        self._freeze(self.expert_freq)
        self._freeze(self.expert_noise_net)

    def forward(self, rgb_mid, diff_seq, prnu_var, fft_var, audio_spec):
        # Inputs:
        # - rgb_mid:  (B, 3, 256, 256) 
        # - diff_seq: (B, 5, 1, 256, 256)
        # - prnu_var: (B, 1, 256, 256) 
        # - fft_var:  (B, 1, 256, 256)
        # - audio_spec: (B, 1, 128, 128) 

        # 0. Router Decision
        weights = self.router(rgb_mid) # (B, 5)
        
        # 1. Temporal (LSTM) Expert
        # Pass the SEQUENCE (diff_seq) directly
        out_temp = torch.sigmoid(self.expert_temp(diff_seq))
        
        # 2. Artifact Expert
        out_art_map = torch.sigmoid(self.expert_art(rgb_mid))
        out_art = out_art_map.flatten(1).max(1)[0].unsqueeze(1)
        
        # 3. Noise (PRNU) Expert
        out_noise = torch.sigmoid(self.expert_noise_head(self.expert_noise_net(prnu_var)))

        # 4. Frequency (FFT) Expert
        out_freq = torch.sigmoid(self.expert_freq(fft_var))
        
        # 5. Audio Expert
        if audio_spec is not None and audio_spec.sum() != 0:
            out_audio = torch.sigmoid(self.expert_audio(audio_spec))
        else:
            out_audio = torch.full_like(out_noise, 0.5)

        # Fusion
        experts = torch.cat([out_temp, out_art, out_noise, out_audio, out_freq], dim=1)
        verdict = (experts * weights).sum(dim=1, keepdim=True)
        
        return verdict, weights

    def _freeze(self, module):
        for param in module.parameters(): param.requires_grad = False
        
    def _load_safe(self, model, path, name):
        try: 
            model.load_state_dict(torch.load(path, weights_only=True))
            print(f">> Loaded {name} Expert.")
        except Exception as e: 
            print(f"!! Failed to load {name} Expert: {e}")
            
    def _load_noise_smart(self, path):
        try:
            state = torch.load(path, weights_only=True)
            if any(k.startswith('net.') for k in state.keys()): self.expert_noise_net.load_state_dict(state, strict=False)
            elif any('prnu_branch' in k for k in state.keys()):
                prnu_state = {k.replace('prnu_branch.net.', 'net.'): v for k, v in state.items() if 'prnu_branch' in k}
                self.expert_noise_net.load_state_dict(prnu_state)
            print(">> Loaded Noise Expert.")
        except: pass
        
def return_model_parameters():
    # print all model parameters and their shapes, indicating which are trainable
    model = MoE_Investigator()
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Trainable Parameter: {name} - {param.shape}")
        else:
            print(f"Frozen Parameter: {name} - {param.shape}")
    
    return

def return_model_summary():
    model = MoE_Investigator()
    print("Model Summary:")
    print(model)
    return