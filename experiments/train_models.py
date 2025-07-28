import argparse
import datetime
import os
import random
import sys
import time
from collections import deque, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pickle
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm, trange

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from torchvision import models, transforms
import torchvision.models.segmentation as segmentation

# Set random seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Allow importing modules from parent directory
sys.path.append("..")

# Custom dataset and dataloader functions
from data_processing.data_processing import ImageLabelDataset, DualImageDataset, create_dataloaders

df = pd.read_pickle("../data/pepper_data.pkl")
def get_dataloader(df, batch_sizes=(32, 64, 64), resize_img_to=(224,224), return_splits=False, double_img=False, transforms=None, num_workers=0):
    """
    for yolo_depthanything model 224,224
    otherwise 512, 288
    LGR had 128,128 
    MobileNetv2 had 224, 224
    """

    # Create domain-specific dataloaders
    domains = df['domain'].unique()
    domain_dataloaders = {}
    test_split_idx = set()
    for domain in domains:
        domain_df = df[df['domain'] == domain]
        #domain_df = domain_df.sample(frac=0.5, random_state=42)
        loaders, split_idx = create_dataloaders(domain_df, batch_sizes=batch_sizes, resize_img_to=resize_img_to, seed=SEED, return_splits=True, double_img=double_img, transforms=transforms, num_workers=num_workers)
        domain_dataloaders[domain] = loaders
        test_split_idx.update(set(split_idx['test']))

    return domain_dataloaders if not return_splits else (domain_dataloaders, test_split_idx)

class NaiveRehearsalBuffer:
    """
    @inproceedings{Hsu18_EvalCL,
        title={Re-evaluating Continual Learning Scenarios: A Categorization and Case for Strong Baselines},
        author={Yen-Chang Hsu and Yen-Cheng Liu and Anita Ramasamy and Zsolt Kira},
        booktitle={NeurIPS Continual learning Workshop },
        year={2018},
        url={https://arxiv.org/abs/1810.12488}
    }
    """
    def __init__(self, buffer_size=1000, balancing=True):
        self.buffer_size = buffer_size
        self.buffer = {}
        self.balancing=balancing

    def update_buffer(self, current_domain, current_dataset):
        # Add/overwrite current domain
        self.buffer[current_domain] = Subset(current_dataset, torch.arange(len(current_dataset)))
        
        # Recalculate quota - even for each domain
        num_domains = len(self.buffer)
        buffer_quota_per_domain = self.buffer_size // num_domains
        
        # Reduce all domains (including current)
        for domain in self.buffer:
            domain_buffer = self.buffer[domain]
            max_safe_samples_to_overwrite = min(buffer_quota_per_domain, len(domain_buffer.dataset))
            rand_indices = torch.randperm(len(domain_buffer.dataset))[:max_safe_samples_to_overwrite].numpy()
            self.buffer[domain] = Subset(domain_buffer.dataset, rand_indices)
    
    def get_loader_with_replay(self, current_domain, current_loader):
        if self.balancing:
            return self.balanced_combine_training_and_replay_loader(current_domain, current_loader)
        else:
            return self.combine_training_and_replay_loader(current_domain, current_loader)
    
    def balanced_combine_training_and_replay_loader(self, current_domain, current_loader):
        current_dataset = current_loader.dataset
        replay_datasets = [dataset for domain, dataset in self.buffer.items() if domain != current_domain]

        if not replay_datasets:
            return current_loader
        
        current_size = len(current_dataset)
        total_replay_size = sum(len(d) for d in replay_datasets)
        samples_per_domain = current_size // len(replay_datasets)
        replay_subsets = []
        
        # Case 1: Small buffer - upsample the buffer - domain stratified upsampling with replacement
        if total_replay_size < current_size:
            for domain_data in replay_datasets:
                indices = torch.randint(0, len(domain_data), (samples_per_domain,))
                replay_subsets.append(Subset(domain_data, indices))
            replay_dataset = ConcatDataset(replay_subsets)
        
        # Case 2: Large buffer - upsample the training data - upsampling with replacement
        elif (total_replay_size > current_size):
            indices = torch.randint(0, current_size, (total_replay_size,))
            current_dataset = Subset(current_dataset, indices)
            replay_dataset = ConcatDataset(replay_datasets)
        
        # Case 3: Large buffer - downsample the buffer - domain stratified downsampling
        # Obsolete - just set the buffer size smaller
        # elif (total_replay_size > current_size) and (self.balancing == 'downsample_buffer'):
        #     for domain_data in replay_datasets:
        #         indices = torch.randperm(len(domain_data))[:samples_per_domain]
        #         replay_subsets.append(Subset(domain_data, indices))
        #     replay_dataset = ConcatDataset(replay_subsets)
        
        # Case 3: Replay buffer = training dataset
        else: 
            replay_dataset = ConcatDataset(replay_datasets)

        combined_dataset = ConcatDataset([replay_dataset, current_dataset])
        combined_loader = DataLoader(
            combined_dataset,
            batch_size=current_loader.batch_size,
            shuffle=True,
            num_workers=current_loader.num_workers,
            pin_memory=current_loader.pin_memory,
            drop_last=current_loader.drop_last
        )
        return combined_loader

    def combine_training_and_replay_loader(self, current_domain, current_loader):
        current_dataset = current_loader.dataset
        replay_datasets = [dataset for domain, dataset in self.buffer.items() if domain != current_domain]

        #Enforces 1:1 ratio when current ≥ buffer
        total_replay = sum(len(dataset) for dataset in replay_datasets)
        if total_replay > 0:
            K = max(len(current_dataset) // total_replay, 1)
            replay_datasets = replay_datasets * K

        combined_dataset =  ConcatDataset(replay_datasets + [current_dataset])
        combined_dataset = DataLoader(
            combined_dataset,
            batch_size=current_loader.batch_size,
            shuffle=True,
            num_workers=current_loader.num_workers,
            pin_memory=current_loader.pin_memory,
            drop_last=current_loader.drop_last
        )
        return combined_dataset
    
    def get_domain_distribution(self):
        """Returns {domain: num_samples} without needing Storage"""
        return {domain: len(subset) for domain, subset in self.buffer.items()}

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

import torch
import torch.nn as nn
import torchvision.models as models

class DualBranchModel(nn.Module):
    def __init__(self, num_outputs=9, dropout_rate=0.3, architecture={'env':'lightweight', 'head':'deep'}):
        super(DualBranchModel, self).__init__()
        self.setup = architecture
        
        self.social_branch = nn.Sequential(
            models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        soc_feature_dim = 1280
        

        if self.setup['env'] == 'lightweight':
            self.env_branch = nn.Sequential(
                models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).features,
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            )
            env_feature_dim = 1280

        elif self.setup['env'] == 'label':
            max_rooms = 30
            self.env_branch = nn.Embedding(max_rooms, 64)
            env_feature_dim = 64


        self.fusion_dim = soc_feature_dim + env_feature_dim

        layers = intermediate_layer_size(soc_feature_dim, num_outputs, 1)
        self.social_classifier = nn.Sequential(
            nn.Linear(soc_feature_dim, layers[0]),
            nn.ReLU(),
            nn.Linear(layers[0], num_outputs)
        )
        layers = intermediate_layer_size(env_feature_dim, num_outputs, 1)
        self.env_classifier = nn.Sequential(
            nn.Linear(env_feature_dim, layers[0]),
            nn.ReLU(),
            nn.Linear(layers[0], num_outputs)
        )
        
        if self.setup['head'] == 'deep':
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
        elif self.setup['head'] == 'shallow':
            self.head = nn.Sequential(
                nn.Linear(self.fusion_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, num_outputs)
            )
        
    def forward(self, social_img, env_img):
        social_features = self.social_branch(social_img)
        env_features = self.env_branch(env_img)
        
        fused_features = torch.cat([social_features, env_features], dim=1)
        scores = self.head(fused_features)

        social_class = self.social_classifier(social_features.detach())
        env_class = self.env_classifier(env_features.detach())
        
        return {
            'output': scores,
            'invariant_domain': social_class,
            'specific_domain': env_class,
            'invariant_feats': social_features,
            'specific_feats': env_features
        }



def heuristic_dualbranch_batch(model, batch, device, detach_base, binary, full_replay, loss_params={}, **kwargs):
    inputs1, inputs2, labels, domain_labels = batch
    inputs1, inputs2, labels, domain_labels = inputs1.to(device), inputs2.to(device), labels.to(device), domain_labels.to(device)
    domain_to_idx = kwargs['domain_to_idx']
    # domain_labels = torch.tensor([domain_to_idx[d] for d in domain_labels], device=device)
    mse_criterion = kwargs['mse_criterion']
    ce_criterion = kwargs['ce_criterion']

    outputs = model(inputs1, inputs2)

    loss = mse_criterion(outputs['output'], labels)
    loss.backward()

    class_optimiser = kwargs['class_optimizer']
    class_optimiser.zero_grad()
    inv_domain_loss = ce_criterion(outputs['invariant_domain'], domain_labels)
    spec_domain_loss = ce_criterion(outputs['specific_domain'], domain_labels)
    (inv_domain_loss + spec_domain_loss).backward()
    class_optimiser.step()
    inv_acc = (outputs['invariant_domain'].argmax(1) == domain_labels).float().mean().item()
    spec_acc = (outputs['specific_domain'].argmax(1) == domain_labels).float().mean().item()
    
    metrics = {
        'inv_domain': inv_domain_loss.item(),
        'spec_domain': spec_domain_loss.item(),
        'inv_acc': inv_acc,
        'spec_acc': spec_acc
    }
    return loss, metrics

#TODO: change dataset to return torch domain labels indexes not strings
def evaluate_model(model, dataloader, criterion, device , tsne=None):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 4:
                inputs1, inputs2, labels, domain_labels = batch
                inputs1 = inputs1.to(device, dtype=torch.float32)
                inputs2 = inputs2.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.float32)
                inputs = (inputs1, inputs2)
            elif len(batch) == 3:
                inputs, labels, _ = batch
                inputs = inputs.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.float32)
                inputs = (inputs,)
            else:
                raise ValueError(f"Batch contains {len(batch)} objects. Should contain 3 or 4 - image/two, labels, domain_labels")

            outputs = model(*inputs)['output']

            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs[0].size(0)
            total_samples += inputs[0].size(0)
            
            if tsne:
                tsne['social'].append(outputs['invariant_feats'].cpu())
                tsne['environmental'].append(outputs['specific_feats'].cpu())
                tsne['domains'].append(domain_labels.cpu())

            val_loss = total_loss / total_samples

    return val_loss if tsne else (val_loss, tsne)

def cross_domain_validation(model, domain_dataloaders, criterion, device, tsne=None):
    results = {}
    for domain, loaders in domain_dataloaders.items():
        val_loader = loaders['val']
        if tsne:
            val_loss, tsne = evaluate_model(model, val_loader, criterion, device, tsne)
        else:
            val_loss = evaluate_model(model, val_loader, criterion, device)
        results[domain] = val_loss
    return results if not tsne else (results, tsne)

def average_metrics(metrics_list):
    # metrics_list: list of dicts, each dict contains metrics for a batch
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    avg_metrics = {}
    for k in keys:
        avg_metrics[k] = float(np.mean([m[k] for m in metrics_list if k in m]))
    return avg_metrics

def collect_tsne_features(model, loader, device):
    model.eval()
    all_inv, all_spec, all_domains = [], [], []
    with torch.no_grad():
        for domain, loaders in domain_dataloaders.items():
            loader = loaders['val']
            for x, _, d in loader:
                x = x.to(device)
                out = model(x)
                all_inv.append(out['invariant_feats'].cpu())
                all_domains += list(d)
    inv_feats = torch.cat(all_inv, dim=0).numpy()
    return inv_feats, all_domains


def collect_gradients(model):
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None and not name.startswith("backbone"):
            module = name.split('.')[0]
            norm = param.grad.norm(2).item()
            if module not in grad_norms:
                grad_norms[module] = []
            grad_norms[module].append(norm)
    # Take mean per module
    grad_norms = {k: float(np.mean(v)) for k, v in grad_norms.items()}
    return grad_norms

import pickle
import torch
from tqdm.notebook import tqdm, trange

def unified_train_loop(
    model, domains, domain_dataloaders, buffer, optimizer, writer, device,
    batch_fn, batch_kwargs, loss_params, num_epochs=5, exp_name="exp", gradient_clipping=False, detach_base=False, binary=False, full_replay=False, collect_tsne_data=False, restart={}, eval_buffer=False
):
    start_domain_idx = 0
    global_step = 0
    history = {
        'train_epoch_loss': [],
        'val_epoch_loss': [],
        'val_buffer_epoch_loss': [],
        'train_epoch_metrics': [],
        'cross_domain_val': [],
        'grad_norms': [],
    }
    
    if restart:
        # Populate history
        global_step = restart['global_step']
        history = restart['history']
        # Populate buffer
        start_domain_idx = np.where(domains == restart['domain'])[0][0]
        for domain_idx, current_domain in enumerate(domains[:start_domain_idx]):
            buffer.update_buffer(current_domain, domain_dataloaders[current_domain]['train'].dataset) 
        print(f"Restarting from domain {restart['domain']} index {start_domain_idx}")
        print(f"Buffer: {buffer.get_domain_distribution()}")         
        

    for domain_idx, current_domain in enumerate(tqdm(domains[start_domain_idx:], desc=f"Total training"), start=start_domain_idx):
        train_loader = buffer.get_loader_with_replay(current_domain, domain_dataloaders[current_domain]['train'])
        if eval_buffer:
            eval_loader = eval_buffer.get_loader_with_replay(current_domain, domain_dataloaders[current_domain]['val'])
        else:
            eval_loader = domain_dataloaders[current_domain]['val']
        len_dataloader = len(train_loader)
        
        for epoch in trange(num_epochs, desc=f"Current domain {current_domain}"):
            model.train()
            epoch_loss = 0.0
            samples = 0
            batch_metrics_list = []
            
            # for batch_idx, batch in enumerate(train_loader):
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Current epoch {epoch}", leave=False)):
                if not batch_kwargs['alpha']:
                    p = (epoch * len_dataloader + batch_idx) / (num_epochs * len_dataloader)
                    alpha = 2. / (1. + np.exp(-10 * p)) - 1
                else:
                    alpha = batch_kwargs['alpha']

                optimizer.zero_grad()
                loss, metrics = batch_fn(model, batch, device, detach_base, binary, full_replay, loss_params, **{**batch_kwargs, 'current_domain': current_domain, 'alpha':alpha})
                if gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                batch_size = batch[0].size(0)
                epoch_loss += loss.item() * batch_size
                samples += batch_size
                global_step += 1
                batch_metrics_list.append(metrics)
                # # TensorBoard logging (every 10 batches)
                # if writer and batch_idx % 10 == 0:
                #     writer.add_scalar(f'{exp_name}/train_loss', loss.item(), global_step)
                #     for k, v in metrics.items():
                #         writer.add_scalar(f'{exp_name}/train_{k}', v, global_step)
            avg_epoch_loss = epoch_loss / samples
            # writer.add_scalar(f'{exp_name}/train_epoch_loss', avg_epoch_loss, global_step)
            history['train_epoch_loss'].append(avg_epoch_loss)
            # Average batch metrics for this epoch
            avg_metrics = average_metrics(batch_metrics_list)
            history['train_epoch_metrics'].append(avg_metrics)

            # Collect gradients
            grad_norms = collect_gradients(model)
            history['grad_norms'].append(grad_norms)

            # Validation on current domain
            val_loss = evaluate_model(model, domain_dataloaders[current_domain]['val'], batch_kwargs['mse_criterion'], device)
            val_loss_buffer = evaluate_model(model, eval_loader, batch_kwargs['mse_criterion'], device)
            # writer.add_scalar(f'{exp_name}/val_epoch_loss', val_loss, global_step)
            history['val_epoch_loss'].append(val_loss)
            history['val_buffer_epoch_loss'].append(val_loss_buffer)

            # Collect data for t-SNE domain separation graphs
            if collect_tsne_data:
                inv_feats, domain_labels = collect_tsne_features(model, domain_dataloaders, device)
                tsne_data = {
                    'inv_feats': inv_feats,
                    'domain_labels': domain_labels
                }
            else:
                tsne_data = None

            # Cross-domain validation (after each domain)
            if epoch == num_epochs-1:
                if collect_tsne_data:
                    tsne = {'social': [], 'env': [], 'domains': []}
                    cross_val, tsne_data = cross_domain_validation(model, domain_dataloaders, batch_kwargs['mse_criterion'], device, tsne)
                else:
                    cross_val = cross_domain_validation(model, domain_dataloaders, batch_kwargs['mse_criterion'], device)
                history['cross_domain_val'].append(cross_val)

                # Only save last model per domain to save space
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'history': history,
                    'tsne' : tsne_data,
                }, f"../checkpoints/{exp_name}_domain{current_domain}_epoch{epoch}_step{global_step}.pt")
            # else:
            #     # Save metrics
            #     torch.save({
            #         # 'model_state_dict': model.state_dict(),
            #         # 'optimizer_state_dict': optimizer.state_dict(),
            #         'history': history,
            #         'tsne' : tsne_data,
            #     }, f"../checkpoints/{exp_name}_domain{current_domain}_epoch{epoch}_step{global_step}.pt")
            with open(f"../checkpoints/{exp_name}_history.pkl", "wb") as f:
                pickle.dump(history, f)
            
        buffer.update_buffer(current_domain, domain_dataloaders[current_domain]['train'].dataset)
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_workers", type=int, required=True)
    args = parser.parse_args()
    num_workers = args.num_workers

    
    model_configs = {
        'heuristic_small_env': (DualBranchModel(), [(288,512), (144,256)], False),
        'heuristic_square_img': (DualBranchModel(), [(224,224)]*2, False),
        'heuristic_eval_buffer': (DualBranchModel(), [(288,512)]*2, True)
    }


    model, img_size, eval_buffer = model_configs[args.model_name]

    transform_soc = transforms.Compose([
        transforms.Resize(img_size[0]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])
    transform_env = transforms.Compose([
        transforms.Resize(img_size[1]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])

    df = pd.read_pickle("../data/pepper_data.pkl")
    df['image_path_env'] = df['image_path'].apply(lambda p: str(Path('../data/masked/environment') / Path(p).name))
    df['image_path_social'] = df['image_path'].apply(lambda p: str(Path('../data/masked/social') / Path(p).name))
    domain_dataloaders = get_dataloader(df, batch_sizes=(16, 64, 64), return_splits=False, double_img=True, transforms=[transform_soc, transform_env], num_workers=num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    domains = df['domain'].unique()
    domain_to_idx = {d: i for i, d in enumerate(domains)}


    dual_model = model.to(device)
    # optimizer = torch.optim.Adam(dual_model.parameters(), lr=1e-3)
    optimizer = optim.Adam([
        {'params': dual_model.social_branch.parameters()},
        {'params': dual_model.env_branch.parameters()},
        {'params': dual_model.head.parameters()},
    ], lr=1e-3)
    classifier_optimizer = optim.Adam([ 
        {'params': dual_model.social_classifier.parameters()},
        {'params': dual_model.env_classifier.parameters()}
    ], lr=1e-3)

    buffer = NaiveRehearsalBuffer(buffer_size=120)
    if eval_buffer:
        eval_buffer = NaiveRehearsalBuffer(buffer_size=120)

    exp_name = f"{args.model_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = None

    dualbranch_kwargs = {
        'mse_criterion': nn.MSELoss(),
        'ce_criterion': nn.CrossEntropyLoss(),
        'domain_to_idx': domain_to_idx,
        'bce_criterion': nn.BCEWithLogitsLoss(),
        'alpha': 0,
        'class_optimizer': classifier_optimizer
    }

    unified_train_loop(
        model=dual_model,
        domains=domains,
        domain_dataloaders=domain_dataloaders,
        buffer=buffer,
        optimizer=optimizer,
        writer=writer,
        device=device,
        batch_fn=heuristic_dualbranch_batch,
        batch_kwargs=dualbranch_kwargs,
        num_epochs=10,
        exp_name=exp_name,
        gradient_clipping=True,
        detach_base=False,
        binary = True,
        full_replay = True,
        collect_tsne_data=False,
        loss_params={'head': 1, 'social': 1, 'room': 0.5},
        eval_buffer=eval_buffer
    )

    # Optional cleanup
    torch.cuda.empty_cache()
    import gc
    gc.collect()

if __name__ == "__main__":
    main()
