import torch
import torch.nn as nn

class DASWeightCNN(nn.Module):
    def __init__(self, spatial_channels=34):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5, 7), padding=(2, 3)),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(16, 32, kernel_size=(3, 5), padding=(1, 2)),
            nn.LeakyReLU(0.1),
            
            # THE FIX: Global Max Pooling
            # Squashes BOTH Spatial (Lanes) and Temporal (Time) to exactly 1.
            # Now it just grabs the absolute loudest peak anywhere in the image!
            nn.AdaptiveMaxPool2d((1, 1)) 
        )
        
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.1),
            # Because we squashed to 1x1, the input is now exactly 32
            nn.Linear(32, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # Keep the absolute value trick so negative waves don't cancel out
        x_abs = torch.abs(x)
        
        features = self.features(x_abs)
        out = self.regressor(features)
        return out