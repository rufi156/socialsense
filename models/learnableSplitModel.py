import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import tqdm, trange
import torch.optim as optim
from torchvision import models
import numpy as np
from collections import deque
import random
import time
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Subset
import torchvision.models.segmentation as segmentation
from collections import defaultdict

#### dualbranch
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GradientReversal(nn.Module):
    def forward(self, x):
        return GradientReversalFunction.apply(x, 1.0)


class DualBranchNet(nn.Module):
    def __init__(self, num_outputs=9, num_domains=6, weights_init=False, weights_norm=False, layer_norm=False, detach_base=True, freeze_base=''):
        super().__init__()

        self.detach_base = detach_base

        def linear(in_f, out_f):
            layer = nn.Linear(in_f, out_f)
            return torch.nn.utils.parametrizations.weight_norm(layer, dim=0) if weights_norm else layer 
        
        def layer_norm(dim):
            return nn.LayerNorm(dim) if layer_norm else nn.Identity()

        self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = 1280

        if freeze_base in ('full', 'partial'):
            for param in self.backbone.parameters():
                param.requires_grad = False
        if freeze_base == 'partial':
            last_layer = list(self.backbone.children())[-1]
            for param in last_layer.parameters():
                param.requires_grad = True

        self.invariant = nn.Sequential(
            linear(self.feature_dim, 256),
            GradientReversal(),
            layer_norm(256),
            nn.ReLU(),
            linear(256, 256)
        )

        self.invariant_domain_classifier = nn.Sequential(
            layer_norm(256),
            nn.Linear(256, num_domains)
        )
        
        self.specific = nn.Sequential(
            linear(self.feature_dim, 256),
            layer_norm(256), 
            nn.ReLU(),
            linear(256, 256)
        )
        
        self.specific_domain_classifier = nn.Sequential(
            layer_norm(256),
            nn.Linear(256, num_domains)
        )

        head_in = 512 + self.feature_dim if self.detach_base else 512
        self.head = nn.Sequential(
            nn.Linear(head_in, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_outputs)
        )

        if weights_init:
            self._init_weights()

    def _init_weights(self):
        custom_modules = [
            'invariant',
            'invariant_domain_classifier',
            'specific',
            'specific_domain_classifier',
            'head'
        ]
        for name in custom_modules:
            module = getattr(self, name, None)
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        base = self.backbone(x)
        base = self.pool(base).view(x.size(0), -1)

        if self.detach_base:
            invariant_feats = self.invariant(base.detach())
            specific_feats = self.specific(base.detach())
            combined = torch.cat([invariant_feats, specific_feats, base], dim=1)
        else:
            invariant_feats = self.invariant(base)
            specific_feats = self.specific(base)
            combined = torch.cat([invariant_feats, specific_feats], dim=1)
        
        invariant_domain_pred = self.invariant_domain_classifier(invariant_feats)
        specific_domain_pred = self.specific_domain_classifier(specific_feats)
        
        scores = self.head(combined)      
        
        return {
            'output': scores,
            'invariant_domain': invariant_domain_pred,
            'specific_domain': specific_domain_pred,
            'invariant_feats': invariant_feats,
            'specific_feats': specific_feats
        }

class DualBranchNet_deep(DualBranchNet):
    """thicker invariant network to learn features instead of realying on the backbone"""
    def __init__(self, num_outputs=9, num_domains=6, weights_init=True, weights_norm=False, layer_norm=False, detach_base=True, freeze_base=''):
        super().__init__(num_outputs, num_domains, weights_init, weights_norm, layer_norm, detach_base, freeze_base)

        def linear(in_f, out_f):
            layer = nn.Linear(in_f, out_f)
            return torch.nn.utils.parametrizations.weight_norm(layer, dim=0) if weights_norm else layer 

        self.invariant = nn.Sequential(
            linear(self.feature_dim, 512),
            nn.ReLU(),
            linear(512, 256),
            nn.ReLU(),
            linear(256, 256)
        )

        self.specific = nn.Sequential(
            linear(self.feature_dim, 512),
            nn.ReLU(),
            linear(512, 256),
            nn.ReLU(),
            linear(256, 256)
        )

        self.invariant_domain_classifier = nn.Sequential(
            GradientReversal(),
            nn.Linear(256, num_domains)
        )
class DualBranchNet_binary(DualBranchNet_deep):
    """binary classifier for the invariant branch"""
    def __init__(self, num_outputs=9, num_domains=6, weights_init=True, weights_norm=False, layer_norm=False, detach_base=False, freeze_base='', explicit_grl=False):
        super().__init__(num_outputs, num_domains, weights_init, weights_norm, layer_norm, detach_base, freeze_base)

        self.explicit_grl = explicit_grl
        
        def gradient_layer():
            return nn.Identity() if explicit_grl else GradientReversal()

        self.invariant_domain_classifier = nn.Sequential(
            gradient_layer(),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        base = self.backbone(x)
        base = self.pool(base).view(x.size(0), -1)

        if self.detach_base:
            invariant_feats = self.invariant(base.detach())
            specific_feats = self.specific(base.detach())
            combined = torch.cat([invariant_feats, specific_feats, base], dim=1)
        else:
            invariant_feats = self.invariant(base)
            specific_feats = self.specific(base)
            combined = torch.cat([invariant_feats, specific_feats], dim=1)
        
        if self.explicit_grl:
            invariant_domain_pred = self.invariant_domain_classifier(GradientReversalFunction.apply(invariant_feats, 1.0))
        else:
            invariant_domain_pred = self.invariant_domain_classifier(invariant_feats)
            
        specific_domain_pred = self.specific_domain_classifier(specific_feats)
        
        scores = self.head(combined)      
        
        return {
            'output': scores,
            'invariant_domain': invariant_domain_pred,
            'specific_domain': specific_domain_pred,
            'invariant_feats': invariant_feats,
            'specific_feats': specific_feats
        }

class DualBranchCNNNet(nn.Module):
    def __init__(self, num_outputs=9, num_domains=6, backbone_type='3conv', branch_type='special', end_type='simple', batch_norm=False, detach_base=False):
        super().__init__()
        self.backbone_type = backbone_type
        self.branch_type = branch_type
        self.end_type = end_type
        self.detach_base = detach_base
        
        def BatchNorm(in_channels):
            return nn.BatchNorm2d(in_channels) if batch_norm else nn.Identity()

        #Input resized to 512,288

        if self.backbone_type == 'none':
            self.backbone = nn.Identity()
            self.backbone_channels = 3

        elif self.backbone_type == '2conv':
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                BatchNorm(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                BatchNorm(64),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            self.backbone_channels = 64

        elif self.backbone_type == '3conv':
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                BatchNorm(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                BatchNorm(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                BatchNorm(128),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            self.backbone_channels = 128

        elif self.backbone_type == 'pretrained':
            self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).features
            if not self.detach_base:
                for param in self.backbone.parameters():
                    param.requires_grad = False
                
            self.backbone_channels = 1280

        else:
            raise ValueError("backbone must be 'none' | '2conv' | '3conv' | 'pretrained'")

        if self.branch_type == 'linear':
            self.social_branch = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(self.backbone_channels, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 256)
            )
            self.room_branch = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(self.backbone_channels, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 256)
            )
            self.branch_channels = 256

        elif self.branch_type == 'simple':
            self.room_branch = nn.Sequential(
                nn.Conv2d(self.backbone_channels, 128, 3, padding=1),
                BatchNorm(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4,4)),
                nn.Flatten()
            )
            self.social_branch = nn.Sequential(
                nn.Conv2d(self.backbone_channels, 128, 3, dilation=2, padding=2),
                BatchNorm(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, 3, padding=1),
                BatchNorm(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4,4)),
                nn.Flatten()
            )
            self.branch_channels = 128*4*4

        elif self.branch_type == 'special':
            self.room_branch = nn.Sequential(
                CoordConv(self.backbone_channels, 128, 3),
                BatchNorm(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4,4)),
                nn.Flatten()
            )
            self.social_branch = nn.Sequential(
                nn.Conv2d(self.backbone_channels, 128, 3, dilation=2, padding=2),
                BatchNorm(128),
                nn.ReLU(),
                SpatialMultiheadAttention(128, 4),
                nn.Conv2d(128, 128, 3, padding=1),
                BatchNorm(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4,4)),
                nn.Flatten()
            )
            self.branch_channels = 128*4*4

        elif self.branch_type == 'adapted_adversarial':
            self.social_branch = nn.Sequential(
                nn.Conv2d(self.backbone_channels, 256, 5, padding=2),
                BatchNorm(256),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(256, 512, 5, padding=2),
                BatchNorm(512),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten()
            )
            self.room_branch = nn.Sequential(
                nn.Conv2d(self.backbone_channels, 256, 5, padding=2),
                BatchNorm(256),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(256, 512, 5, padding=2),
                BatchNorm(512),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten()
            )
            # Calculate output dimensions based on backbone output            
            adversarial_channels = {
                'none': 512 * 128 * 72,       # 4,718,592
                '2conv': 512 * 32 * 18,       # 294,912
                '3conv': 512 * 16 * 9,        # 73,728
                'pretrained': 512 * 4 * 2     # 4,096
            }
            self.branch_channels = adversarial_channels[self.backbone_type]

        elif self.branch_type == 'adversarial':
            self.social_branch = nn.Sequential(
                nn.Conv2d(self.backbone_channels, 96, 5, padding=2),
                BatchNorm(96),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(96, 144, 3, padding=1),
                BatchNorm(144),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(144, 256, 5, padding=2),
                BatchNorm(256),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten()
            )
            self.room_branch = nn.Sequential(
                nn.Conv2d(self.backbone_channels, 96, 5, padding=2),
                BatchNorm(96),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(96, 144, 3, padding=1),
                BatchNorm(144),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(144, 256, 5, padding=2),
                BatchNorm(256),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten()
            )
            # Calculate output dimensions based on backbone output
            adversarial_channels = {
                'none': 256 * 64 * 36,        # 589,824
                '2conv': 256 * 16 * 9,        # 36,864 
                '3conv': 256 * 8 * 4,         # 8,192
                'pretrained': 256 * 2 * 1     # 512
            }
            self.branch_channels = adversarial_channels[self.backbone_type]

        else:
            raise ValueError("branch must be 'linear' | 'simple' | 'special' | 'adversarial' | 'adapted_adversarial'")

        if self.end_type == 'simple':
            self.room_domain_cls = nn.Linear(self.branch_channels, num_domains)
            self.social_domain_cls = nn.Linear(self.branch_channels, 1)
            self.head = nn.Sequential(
                nn.Linear(self.branch_channels*2, 256),
                nn.ReLU(),
                nn.Linear(256, num_outputs)
            )
        elif self.end_type == 'adversarial':
            self.room_domain_cls = nn.Sequential(
                nn.Linear(self.branch_channels, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, num_domains)
            )
            self.social_domain_cls = nn.Sequential(
                nn.Linear(self.branch_channels, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1)
            )
            self.head = nn.Sequential(
                nn.Linear(self.branch_channels*2, 512),
                nn.ReLU(),
                nn.Linear(512, num_outputs)
            )
        
        elif self.end_type == '3linear':
            self.room_domain_cls = nn.Sequential(
                nn.Linear(self.branch_channels, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, num_domains)
            )
            self.social_domain_cls = nn.Sequential(
                nn.Linear(self.branch_channels, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )
            if self.detach_base:
                self.backbone_proj = nn.Sequential(
                    nn.AdaptiveAvgPool2d((4,4)),
                    nn.Flatten(),
                    nn.Linear(self.backbone_channels*4*4, self.branch_channels),
                    nn.ReLU(),
                )
                proj_backbone_channels = self.branch_channels
            total_channels = self.branch_channels*2
            total_channels += proj_backbone_channels if self.detach_base else 0
            self.head = nn.Sequential(
                nn.Linear(total_channels, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, num_outputs)
            )
        else:
            raise ValueError("end must be 'simple' | '3linear' | 'adversarial'")

    def forward(self, x, alpha=1.0):
        base = self.backbone(x)
        
        if self.detach_base:
            room_feat = self.room_branch(base.detach())
            social_feat = self.social_branch(base.detach())
            proj_base = self.backbone_proj(base)
            scores = self.head(torch.cat([room_feat, social_feat, proj_base], 1))
        else:
            room_feat = self.room_branch(base)
            social_feat = self.social_branch(base)
            scores = self.head(torch.cat([room_feat, social_feat], 1))

        room_domain_cls = self.room_domain_cls(room_feat)
        social_domain_cls = self.social_domain_cls(GradientReversalFunction.apply(social_feat, alpha))
        
        return {
            'output': scores,
            'invariant_domain': social_domain_cls,
            'specific_domain': room_domain_cls,
            'invariant_feats': social_feat,
            'specific_feats': room_feat
        }
        
class DualBranchNet_DANN(nn.Module):
    def __init__(self, num_outputs=9, backbone_type='3conv'):
        super().__init__()
        self.backbone_type = backbone_type

        #Input RGB of size 512,288

        if self.backbone_type == 'mobilenet':
            self.social_branch = nn.Sequential(
                models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).features,
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
            self.backbone_channels = 1280

        elif self.backbone_type == '3conv':
            self.social_branch = nn.Sequential(
                nn.Conv2d(3, 96, 5, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(96, 144, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(144, 256, 5, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d(output_size=(4, 4)),
                nn.Flatten()
            )
            self.backbone_channels = 256*4*4
        else:
            raise ValueError("backbone_type must be '3conv' | 'mobilenet'")

        self.social_domain_cls = nn.Sequential(
            nn.Linear(self.backbone_channels, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )

        self.head = nn.Sequential(
            nn.Linear(self.backbone_channels, 512),
            nn.ReLU(),
            nn.Linear(512, num_outputs)
        )

    def forward(self, x, alpha=1.0, is_first_domain=False):
        social_feat = self.social_branch(x)
        scores = self.head(social_feat)
        
        social_domain_cls = None if is_first_domain else self.social_domain_cls(GradientReversalFunction.apply(social_feat, alpha))
        
        return {
            'output': scores,
            'invariant_domain': social_domain_cls,
            'invariant_feats': social_feat
        }

class CoordConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels + 2, 
                             out_channels, 
                             kernel_size, 
                             padding=kernel_size//2)

    def forward(self, x):
        batch, _, h, w = x.shape
        
        # Create coordinate grids (range [-1, 1])
        x_coord = torch.linspace(-1, 1, w).repeat(h, 1)
        y_coord = torch.linspace(-1, 1, h).repeat(w, 1).t()
        coords = torch.stack([x_coord, y_coord], dim=0)
        coords = coords.unsqueeze(0).repeat(batch, 1, 1, 1).to(x.device)
        x = torch.cat([x, coords], dim=1)
        
        return self.conv(x)

class SpatialMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x_small = self.pool(x)
        # Reshape to sequence format
        x_flat = x_small.flatten(2).permute(0, 2, 1)
        attn_output, _ = self.mha(x_flat, x_flat, x_flat)
        # Reshape back to spatial format
        attn_reshaped = attn_output.permute(0, 2, 1).view(B, C, 8, 8)
        return F.interpolate(attn_reshaped, size=(H, W), mode='bilinear')
    
class DualBranchNet_minimal(nn.Module):
    def __init__(self, num_outputs=9, backbone_type='3conv', branch_type='simple', head_type='3layer', batch_norm=False):
        super().__init__()
        self.backbone_type = backbone_type
        self.branch_type = branch_type
        self.head_type = head_type
        
        def BatchNorm(in_channels):
            return nn.BatchNorm2d(in_channels) if batch_norm else nn.Identity()

        #Input resized to 512,288
        if self.backbone_type == 'absurd':
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            ) 

        elif self.backbone_type == '3conv':
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                BatchNorm(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                BatchNorm(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                BatchNorm(128),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            self.backbone_channels = 128

        elif self.backbone_type == 'pretrained':
            self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).features
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone_channels = 1280

        if self.branch_type == 'simple':
            self.social_branch = nn.Sequential(
                nn.Conv2d(self.backbone_channels, 128, 3, dilation=2, padding=2),
                BatchNorm(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, 3, padding=1),
                BatchNorm(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4,4)),
                nn.Flatten()
            )

        elif self.branch_type == 'special':
            self.social_branch = nn.Sequential(
                nn.Conv2d(self.backbone_channels, 128, 3, dilation=2, padding=2),
                BatchNorm(128),
                nn.ReLU(),
                SpatialMultiheadAttention(128, 4),
                nn.Conv2d(128, 128, 3, padding=1),
                BatchNorm(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4,4)),
                nn.Flatten()
            )
        elif self.branch_type == '1convo':
            self.social_branch = nn.Sequential(
                nn.Conv2d(self.backbone_channels, 128, 3, padding=1),
                BatchNorm(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4,4)),
                nn.Flatten()
            )
        elif self.branch_type == 'absurd':
            self.social_branch = nn.Sequential(
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )

        if self.head_type == '3layer':
            self.head = nn.Sequential(
                nn.Linear(128*4*4, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, num_outputs)
            )
        elif self.head_type == '2layer256':
            self.head = nn.Sequential(
                nn.Linear(128*4*4, 256),
                nn.ReLU(),
                nn.Linear(256, num_outputs)
            )
        elif self.head_type == 'absurd':
            self.head = nn.Sequential(
                nn.Linear(32, num_outputs)
            )
    

    def forward(self, x, alpha=1.0):
        base = self.backbone(x)
    
        social_feat = self.social_branch(base)
        scores = self.head(social_feat)
        
        return {
            'output': scores,
            'invariant_feats': social_feat,
        }
        
import torchvision.models.segmentation as segmentation

class DANN_classifier_poc(nn.Module):
    def __init__(self):
        super().__init__()
        pretrained_model = segmentation.deeplabv3_mobilenet_v3_large(
            pretrained=True,
            weights=segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        )
        self.backbone = pretrained_model.backbone
        self.classifier = pretrained_model.classifier
        self.bn = nn.BatchNorm1d(4116)

        self.social_domain_cls = nn.Sequential(
            nn.Linear(4116, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )
        self.head = nn.Sequential(
            nn.Linear(4116, 512),
            nn.ReLU(),
            nn.Linear(512, 9)
        )

    def forward(self, x, alpha=1.0, is_first_domain=False):
        features = self.backbone(x)['out']
        features = self.classifier(features)
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)
        features = self.bn(features)

        domain = None if is_first_domain else self.social_domain_cls(GradientReversalFunction.apply(features, alpha))
        output = self.head(features)
        return {
            'output': output,
            'invariant_domain': domain,
            'invariant_feats': features,
        }
