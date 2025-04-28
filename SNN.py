import torch
import torch.nn as nn

# ----------------------------------------
# Model Definitions (for 64×64 inputs)
# ----------------------------------------
class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            # in: 1×64×64
            nn.Conv2d(1, 64, kernel_size=10),     # → 64×(64−10+1)=55
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                      # → 64×27×27

            nn.Conv2d(64, 128, kernel_size=7),    # → 128×21×21
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                      # → 128×10×10

            nn.Conv2d(128, 128, kernel_size=4),   # → 128×7×7
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                      # → 128×3×3

            # pad=1 so 3×3 → (3+2−4+1)=2×2
            nn.Conv2d(128, 256, kernel_size=4, padding=1),
            nn.ReLU(inplace=True),

            # now spatial is 256×2×2 → keep as is
            nn.AdaptiveAvgPool2d((2, 2))
        )
        self.fc = nn.Sequential(
            nn.Flatten(),                         # → 256*2*2 = 1024
            nn.Linear(256 * 2 * 2, 4096),
            nn.Sigmoid()
        )
    
    def forward_once(self, x):
        x = self.cnn(x)
        return self.fc(x)
    
    def forward(self, img1, img2):
        return self.forward_once(img1), self.forward_once(img2)


class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, embed1, embed2, label):
        # Euclidean distance
        dist = torch.norm(embed1 - embed2, p=2, dim=1)
        # positive pairs
        loss_pos = label * dist.pow(2)
        # negative pairs
        loss_neg = (1 - label) * torch.clamp(self.margin - dist, min=0).pow(2)
        return torch.mean(loss_pos + loss_neg)
