import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

class HybridDRClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super(HybridDRClassifier, self).__init__()
        
        # 1. Vision Transformer Pathway
        self.vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        vit_out_dim = self.vit.heads.head.in_features
        self.vit.heads = nn.Identity()  # Remove original classification head
        
        # 2. ResNet50 CNN Pathway
        from torchvision.models import resnet50, ResNet50_Weights
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1]) # Remove fc layer, outputs [B, 2048, 1, 1]
        cnn_out_dim = 2048
        
        # 3. Fusion & Classification Head
        self.fc = nn.Sequential(
            nn.Linear(vit_out_dim + cnn_out_dim, 512),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, img_vit, img_cnn=None):
        if img_cnn is None: 
            img_cnn = img_vit
            
        f_vit = self.vit(img_vit)
        
        f_cnn_spatial = self.cnn(img_cnn)
        f_cnn = f_cnn_spatial.view(f_cnn_spatial.size(0), -1) 
        
        f_fused = torch.cat((f_vit, f_cnn), dim=1)
        logits = self.fc(f_fused)
        
        return logits, f_fused
