import torch
import torch.nn as nn
from tqdm.notebook import tqdm, trange
import torch.optim as optim
from torchvision import models
import numpy as np

#### baselines
class SocialContinualModel(nn.Module):
    def __init__(self, num_tasks=9):
        super().__init__()
        # Initialize the ResNet50 architecture with Places365 configuration
        self.backbone = models.resnet50(num_classes=365)
        
        # get Places365 weights and fix their naming leftover from troch saving convention
        places365_weights = torch.load('resnet50_places365.pth.tar', weights_only=True)
        state_dict = places365_weights['state_dict']
        state_dict = {k.replace('module.', ''): v 
                     for k, v in state_dict.items()}
        
        # Load weights
        self.backbone.load_state_dict(state_dict)
        
        # Remove classification head
        self.backbone.fc = nn.Identity()

        #Freeze all params except last layer
        for name, param in self.backbone.named_parameters():
            if 'layer4' not in name:
                param.requires_grad_(False)
        
        #TODO human mask average, std, quadrants, human in realtion to robot std
        


        # Shared layers #TODO deeper shared space?
        self.shared_fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Task-specific heads with more expensive but finegrained GELU
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1024, 512),
                nn.GELU(),
                nn.Linear(512, 1),
                nn.Sigmoid()
            ) for _ in range(num_tasks)
        ])

    def forward(self, x):
        features = self.backbone(x)
        shared = self.shared_fc(features)
        outputs = [head(shared) for head in self.heads]
        return torch.cat(outputs, dim=1)

import torch.nn as nn
from torchvision import models

class LGRBaseline(nn.Module):
    """
    @misc{churamani_feature_2024,
		title = {Feature Aggregation with Latent Generative Replay for Federated Continual Learning of Socially Appropriate Robot Behaviours},
		url = {http://arxiv.org/abs/2405.15773},
		doi = {10.48550/arXiv.2405.15773},
		number = {{arXiv}:2405.15773},
		publisher = {{arXiv}},
		author = {Churamani, Nikhil and Checker, Saksham and Chiang, Hao-Tien Lewis and Gunes, Hatice},
		urldate = {2025-01-30},
		date = {2024-03-16},
	}
    """
    def __init__(self, num_classes=9):
        super(LGRBaseline, self).__init__()
        
        # MobileNetV2 backbone
        self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).features
        
        # Backbone feature processing
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        
        # Regression head
        self.fc1_bn = nn.BatchNorm1d(1280)
        self.fc2 = nn.Linear(1280, 32)
        # self.fc2_bn = nn.BatchNorm1d(128)
		# self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, num_classes)

    def forward(self, x):
        # Feature extraction
        x = self.backbone(x)
        
        # Spatial reduction
        x = self.pool(x)
        x = self.flatten(x)
        
        # Regression
        x = self.fc1_bn(x)
        x = self.fc2(x)
        # x = self.fc2_bn(x)
		# x = self.fc3(x)
        x = self.fc4(x)
        
        return {'output': x}
