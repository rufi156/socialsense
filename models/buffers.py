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
import pandas as pd

class ReservoirBuffer:
    def __init__(self, capacity=1000, replay_ratio=0.2, input_shape=(3, 384, 216), label_shape=(9,), device=torch.device('cpu')):
        self.capacity = capacity
        self.inputs = torch.empty((capacity, *input_shape), dtype=torch.float32, device=device)
        self.labels = torch.empty((capacity, *label_shape), dtype=torch.float32, device=device)
        self.domains = [None] * capacity
        self.size = 0          # Number of samples currently in buffer
        self.num_seen = 0      # Total samples seen
        self.replay_ratio = replay_ratio

    def add(self, new_samples):
        for sample in new_samples:
            self.num_seen += 1
            if self.size < self.capacity:
                idx = self.size
                self.size += 1
            else:
                idx = random.randint(0, self.num_seen - 1)
                if idx >= self.capacity:
                    continue
            self.inputs[idx].copy_(sample[0])
            self.labels[idx].copy_(sample[1])
            self.domains[idx] = sample[2]

    def sample(self, batch_size):
        if self.size == 0:
            return []
        indices = torch.randint(0, self.size, (batch_size,))
        return [(self.inputs[i], self.labels[i], self.domains[i]) for i in indices]

    def __len__(self):
        return self.size

    def get_domain_distribution(self):
        return pd.Series(self.domains[:self.size]).value_counts()

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

class NonstratifiedNaiveRehearsalBuffer:
    def __init__(self, buffer_size=1000):
        self.buffer_size = buffer_size
        self.buffer = []  # List of (dataset_object, idx) tuples

    def update_buffer(self, current_dataset):
        # Add all new samples as (dataset_object, idx) pairs
        new_samples = [(current_dataset, idx) for idx in range(len(current_dataset))]
        self.buffer += new_samples

        # If buffer is too large, randomly keep only buffer_size samples
        if len(self.buffer) > self.buffer_size:
            perm = torch.randperm(len(self.buffer))[:self.buffer_size]
            self.buffer = [self.buffer[i] for i in perm]

    def get_loader_with_replay(self, current_loader):
        current_dataset = current_loader.dataset

        # Group buffer samples by dataset for efficient Subset creation       
        dataset_to_indices = defaultdict(list)
        for dataset, idx in self.samples:
            dataset_to_indices[dataset].append(idx)
        buffer_subsets = [Subset(ds, idxs) for ds, idxs in dataset_to_indices.items()]
        
        if not buffer_subsets:
            return current_loader
        
        replay_dataset = ConcatDataset(buffer_subsets)

        current_size = len(current_dataset)
        total_replay_size = len(replay_dataset)

        N_max = max(total_replay_size, current_size)

        # Upsample with replacement if needed
        if total_replay_size < current_size:
            idxs = torch.randint(0, total_replay_size, (current_size,))
            replay_dataset = Subset(replay_dataset, idxs)
        if total_replay_size > current_size:
            idxs = torch.randint(0, current_size, (total_replay_size,))
            current_dataset = Subset(current_dataset, idxs)

        combined_dataset = ConcatDataset([replay_dataset, current_dataset])
        combined_loader =  DataLoader(
            combined_dataset,
            batch_size=current_loader.batch_size,
            shuffle=True,
            num_workers=current_loader.num_workers,
            pin_memory=current_loader.pin_memory,
            drop_last=current_loader.drop_last
        )
        return combined_loader


    def __len__(self):
        return len(self.buffer)
    
    def get_domain_distribution(self):
        return {"buffer": len(self.buffer)}
