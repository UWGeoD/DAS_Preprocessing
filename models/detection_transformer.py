# models/transformer.py
import torch
import torch.nn as nn

class DASCountTransformer(nn.Module):
    def __init__(self, spatial_channels=34, d_model=64, nhead=4, num_layers=2, num_classes=6):
        super().__init__()
        
        # Compress the spatial dimension (channels) into a feature vector per time step
        # Input shape expected: (Batch, 1, Channels, Time)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=(spatial_channels, 5), padding=(0, 2)),
            nn.BatchNorm2d(d_model),
            nn.ReLU(),
        )
        
        # Transformer looks across the time sequence
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Head 1: Count Predictor (Regression)
        self.count_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Head 2: Vehicle Type Predictor (Classification)
        self.type_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes) # Outputs 6 raw scores
        )

    def forward(self, x):
        # x: (B, 1, C, T)
        x = self.spatial_conv(x)  # -> (B, d_model, 1, T)
        
        # Squeeze out the spatial dim and swap axes for the transformer: (B, T, d_model)
        x = x.squeeze(2).permute(0, 2, 1) 
        
        # Pass through transformer
        x = self.transformer(x)
        
        # Global average pooling over time
        x = x.mean(dim=1) # -> (B, d_model)
        
        # Branch off to the two heads
        count_pred = self.count_head(x)
        type_logits = self.type_head(x)
        
        return count_pred, type_logits