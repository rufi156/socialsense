import pandas as pd
import torch
from itertools import product
from collections import Counter
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
import numpy as np
import random
from collections import Counter
from tqdm import tqdm
import json
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

import sys
sys.path.append('..')

from data_processing.data_processing import create_dataloaders

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
    def __init__(self, buffer_size=1000, balancing=''):
        """
        balancing options: downsample_buffer | upsample_current
        """
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
        
        # Case 2: Large buffer - downsample the buffer - domain stratified downsampling
        elif (total_replay_size > current_size) and (self.balancing == 'downsample_buffer'):
            for domain_data in replay_datasets:
                indices = torch.randperm(len(domain_data))[:samples_per_domain]
                replay_subsets.append(Subset(domain_data, indices))
            replay_dataset = ConcatDataset(replay_subsets)

        # Case 3: Large buffer - upsample the training data - upsampling with replacement
        elif (total_replay_size > current_size) and (self.balancing == 'upsample_current'):
            indices = torch.randint(0, current_size, (total_replay_size,))
            current_dataset = Subset(current_dataset, indices)
            replay_dataset = ConcatDataset(replay_datasets)
        
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




def get_dataloader(df, batch_sizes=(32, 64, 64), resize_img_to=(224,224), return_splits=False, double_img=False, transforms=None):
    """
    for yolo_depthanything model 224,224
    otherwise 512, 288
    LGR had 128,128 
    MobileNetv2 had 224, 224
    """
    df = pd.read_pickle("../data/pepper_data.pkl")
    # Create domain-specific dataloaders
    domains = df['domain'].unique()
    domain_dataloaders = {}
    test_split_idx = set()
    for domain in domains:
        domain_df = df[df['domain'] == domain]
        #domain_df = domain_df.sample(frac=0.5, random_state=42)
        loaders, split_idx = create_dataloaders(domain_df, batch_sizes=batch_sizes, resize_img_to=resize_img_to, seed=SEED, return_splits=True, double_img=double_img, transforms=transforms)
        domain_dataloaders[domain] = loaders
        test_split_idx.update(set(split_idx['test']))

    return domain_dataloaders if not return_splits else (domain_dataloaders, test_split_idx)


def main():
    df = pd.read_pickle("../data/pepper_data.pkl")
    domain_dataloaders = get_dataloader(df, batch_sizes=(32, 64, 64), resize_img_to=(288,512), return_splits=False, double_img=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    domains = df['domain'].unique()

    versions = list(product([1000, 500, 200, 100], ['', 'downsample_buffer', 'upsample_current']))

    results = {}

    for buffer_size, balanced in tqdm(versions):
        key = f"{buffer_size}_{balanced or 'standard'}"
        results[key] = {}
        buffer = NaiveRehearsalBuffer(buffer_size, balanced)
        for current_domain in tqdm(domains, leave=False):
            train_loader = buffer.get_loader_with_replay(current_domain, domain_dataloaders[current_domain]['train'])
            ratio = Counter()
            for _, _, labels in tqdm(train_loader, leave=False):
                ratio.update(labels)  # labels is a list
            results[key][current_domain] = dict(ratio)
            buffer.update_buffer(current_domain, domain_dataloaders[current_domain]['train'].dataset)

    with open("buffer_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()