import torch
import torchvision
from torch import nn

class OneHeadModel(nn.Module):
    def __init__(self, device, p_dropout):
        super(OneHeadModel, self).__init__()

        self.device = device
        self.p_dropout = p_dropout

        # Load EfficientNet encoder
        # weights = torchvision.models.EfficientNet_B1_Weights.DEFAULT
        # efficientNet = torchvision.models.efficientnet_b1(weights=weights)
        # self.encoder = efficientNet.features

        # Load EfficientNet encoder
        denseNet = torchvision.models.densenet121(weights='DEFAULT')
        self.encoder = denseNet.features

        # Pooling layers
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Classification head
        self.classification_head = nn.Sequential(
            nn.Dropout(p=self.p_dropout),
            nn.Linear(1024, 5) # 5 output nodes for classification
            )     

    def forward(self, x):
        x = self.encoder(x) # Extract features

        # Apply pooling layers
        enc_out = self.global_avg_pool(x).view(x.size(0), -1)

        # Classification branch
        class_out = self.classification_head(enc_out).float()

        return class_out, enc_out

    