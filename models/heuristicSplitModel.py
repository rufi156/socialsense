import torch
import torch.nn as nn
import torchvision.models as models
import copy
import warnings

def intermediate_layer_size(input, output, n_layers):
    start_exp = (output + 1).bit_length()
    end_exp = (input - 1).bit_length()
    
    total_powers = end_exp - start_exp
    if total_powers < n_layers:
        return None

    result = []
    denominator = n_layers + 1
    half_denominator = denominator // 2

    for i in range(1, n_layers + 1):
        numerator = i * total_powers + half_denominator #works same as rounding
        idx = numerator // denominator
        power = 1 << (start_exp + idx)
        result.append(power)

    return result

class CLIPEncoderWrapper(nn.Module):
    """Wrapper for the forward .encode_image() function of the CLIP encoder"""
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        for p in self.clip_model.parameters():
            p.requires_grad = False
    def forward(self, x):
        with torch.no_grad():
            return self.clip_model.encode_image(x)

class DualBranchModel(nn.Module):
    def __init__(self, num_outputs=9, dropout_rate=0.3, setup={'branch':'mobilenetv2'}, clip_model=None, freeze_branches=False, branch_norm=False):
        assert not (setup['branch'] == 'clip' and clip_model is None), "clip_model must be provided for CLIP branch"
        if setup['branch'] == 'clip' and freeze_branches is False:
            warnings.warn("CLIP branch will be frozen regardless of freeze_branches", UserWarning)
        super(DualBranchModel, self).__init__()
        self.setup = setup
        self.freeze_branches = freeze_branches
        self.branch_norm = branch_norm

        if self.setup['branch'] == 'resnet18':
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            branch = nn.Sequential(
                *list(model.children())[:-2],
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            )
            branch_feature_dim = 512
        elif self.setup['branch'] == 'mobilenetv2':
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).features
            branch = nn.Sequential(
                model,
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            )
            branch_feature_dim = 1280
        elif self.setup['branch'] == 'efficientnetb0':
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1).features
            branch = nn.Sequential(
                model,
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            )
            branch_feature_dim = 1280
        elif self.setup['branch'] == 'clip':
            branch = CLIPEncoderWrapper(clip_model)
            branch_feature_dim = 512
        
        elif self.setup['branch'] == 'simple':
            branch = nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(8, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            )
            branch_feature_dim = 64

        # Optional freeze params
        if self.freeze_branches == 'full':
            for param in branch.parameters():
                param.requires_grad = False
        if self.freeze_branches == 'partial':
            for param in branch[:15].parameters():
                param.requires_grad = False

        
        if self.branch_norm:
            branch = branch.append(nn.LayerNorm(branch_feature_dim))
        
        # If using frozen CLIP reusing the model is more efficient
        if self.setup['branch'] == 'clip':
            self.social_branch = branch
            self.env_branch = branch
        else:
            self.social_branch = branch
            self.env_branch = copy.deepcopy(branch)

        soc_feature_dim = branch_feature_dim
        env_feature_dim = branch_feature_dim

        # Override one of the branches to run ablations
        if self.setup.get('env') == 'ablated':
            env_feature_dim = 0


        self.fusion_dim = soc_feature_dim + env_feature_dim
        
        self.head = nn.Sequential(
            nn.Linear(self.fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, num_outputs)
        )
        
    def forward(self, social_imgs, env_imgs=None):
        if env_imgs is None:
            env_features = torch.zeros(social_imgs.size(0), 0, device=social_imgs.device, dtype=social_imgs.dtype)
        else:
            env_features = self.env_branch(env_imgs)
        
        social_features = self.social_branch(social_imgs)

        fused_features = torch.cat([social_features, env_features], dim=1)
        scores = self.head(fused_features)

        return {
            'output': scores,
            'invariant_feats': social_features,
            'specific_feats': env_features
        }