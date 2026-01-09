import torch
import torch.nn as nn
import torchvision.models as models

class SkinCancerClassifier(nn.Module):
    def __init__(self, pretrained=True):
        super(SkinCancerClassifier, self).__init__()
        # Load pretrained ResNet18
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # Replace final layer for binary classification
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 1)
        )
    
    def forward(self, x):
        return self.resnet(x)
