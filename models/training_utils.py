import torch
from tqdm.notebook import tqdm, trange
import numpy as np
import pickle
import datetime
import os
import torch
from typing import Dict, Any, Tuple
TQDM_DISABLED = bool(os.environ.get("TQDM_DISABLE", "0") == "1")

### setup
# For LGRBaseline
def baseline_batch(model, batch, device, detach_base, binary, full_replay, loss_params, **kwargs):
    inputs, labels, _ = batch
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)['output']
    loss = kwargs['mse_criterion'](outputs, labels)
    loss.backward()
    metrics = {}
    return loss, metrics

#For DANN
def dann_batch(model, batch, device, detach_base, binary, full_replay, loss_params={'head':0.5, 'social':0.5}, **kwargs):
    inputs, labels, domain_labels = batch
    inputs, labels = inputs.to(device), labels.to(device)
    domain_to_idx = kwargs['domain_to_idx']
    domain_labels = torch.tensor([domain_to_idx[d] for d in domain_labels], device=device)
    mse_criterion = kwargs['mse_criterion']
    bce_criterion = kwargs['bce_criterion']
    alpha = kwargs['alpha']

    current_domain = kwargs['current_domain']
    current_binary_labels = (domain_labels == domain_to_idx[current_domain]).float()
    is_first_domain = bool(domain_to_idx[current_domain] == 0)

    outputs = model(inputs, alpha=alpha, is_first_domain=is_first_domain)

    task_loss = mse_criterion(outputs['output'], labels)

    if is_first_domain:
        inv_domain_loss = 0
        inv_acc = 0
    else:
        inv_domain_loss = bce_criterion(outputs['invariant_domain'].squeeze(), current_binary_labels)
        preds = (outputs['invariant_domain'].squeeze() > 0).float()
        inv_acc = (preds == current_binary_labels).float().mean().item()
    
    total_loss = (loss_params['head'] * task_loss +
                    loss_params['social'] * inv_domain_loss)

    total_loss.backward()
    
    metrics = {
        'task_loss': task_loss.item(),
        'inv_domain': 0 if is_first_domain else inv_domain_loss.item(),
        'inv_acc': inv_acc
    }
    return total_loss, metrics

def heuristic_dualbranch_batch(model, batch, device, **kwargs):
    if len(batch) == 4:
        inputs1, inputs2, labels, domain_labels = batch
        inputs1, inputs2, labels, domain_labels = inputs1.to(device), inputs2.to(device), labels.to(device), domain_labels.to(device)
        inputs = (inputs1, inputs2)
    elif len(batch) == 3:
        inputs1, labels, domain_labels = batch
        inputs1, labels, domain_labels = inputs1.to(device), labels.to(device), domain_labels.to(device)
        inputs = (inputs1,)
    else:
        raise ValueError(f"Batch contains {len(batch)} objects. Should contain 3 or 4 - image/two, labels, domain_labels")

    mse_criterion = kwargs['mse_criterion']

    outputs = model(*inputs)

    loss = mse_criterion(outputs['output'], labels)
    
    metrics = {}
    return loss, metrics


# For DualBranchNet
def dualbranch_batch(model, batch, device, detach_base, binary, full_replay, loss_params={'head': 1, 'social': 0.5, 'room': 0.2}, **kwargs):
    inputs, labels, domain_labels = batch
    inputs, labels = inputs.to(device), labels.to(device)
    domain_to_idx = kwargs['domain_to_idx']
    domain_labels = torch.tensor([domain_to_idx[d] for d in domain_labels], device=device)
    mse_criterion = kwargs['mse_criterion']
    ce_criterion = kwargs['ce_criterion']
    cos_criterion = kwargs['cos_criterion']
    alpha = kwargs['alpha']
    if binary:
        bce_criterion = kwargs['bce_criterion']

    # Split batch
    current_domain = kwargs['current_domain']
    current_binary_labels = (domain_labels == domain_to_idx[current_domain]).float()
    current_mask = (domain_labels == domain_to_idx[current_domain])

    if full_replay:
        current_mask = torch.ones_like(current_mask, dtype=torch.bool)

    replay_mask = ~current_mask

    # 1. Current samples: update all parameters
    if current_mask.any():
        inputs_current = inputs[current_mask]
        labels_current = labels[current_mask]
        domain_labels_current = domain_labels[current_mask]

        outputs_current = model(inputs_current, alpha=alpha)
        inv_feats = outputs_current['invariant_feats']
        spec_feats = outputs_current['specific_feats']

        task_loss = mse_criterion(outputs_current['output'], labels_current)
        if binary:
            inv_domain_loss = bce_criterion(outputs_current['invariant_domain'].squeeze(), current_binary_labels[current_mask])
        else:
            inv_domain_loss = ce_criterion(outputs_current['invariant_domain'], domain_labels_current)
        spec_domain_loss = ce_criterion(outputs_current['specific_domain'], domain_labels_current)
        similarity_loss = cos_criterion(inv_feats, spec_feats)
        
        total_loss = (loss_params['head'] * task_loss +
                      loss_params['social'] * inv_domain_loss +
                      loss_params['room'] * spec_domain_loss)

        total_loss.backward(retain_graph= not full_replay)
        
        if binary:
            # Threshold at 0 (sigmoid(0) = 0.5)
            preds = (outputs_current['invariant_domain'].squeeze() > 0).float()
            inv_acc = (preds == current_binary_labels[current_mask]).float().mean().item()
        else:
            inv_acc = (outputs_current['invariant_domain'].argmax(1) == domain_labels_current).float().mean().item()
        spec_acc = (outputs_current['specific_domain'].argmax(1) == domain_labels_current).float().mean().item()
    else:
        total_loss = torch.tensor(0.0, device=device)
        inv_acc = 0.0
        spec_acc = 0.0
        task_loss = torch.tensor(0.0, device=device)
        inv_domain_loss = torch.tensor(0.0, device=device)
        spec_domain_loss = torch.tensor(0.0, device=device)
        similarity_loss = torch.tensor(0.0, device=device)

    # 2. Replay samples: update only specific branch + head
    if replay_mask.any():
        inputs_replay = inputs[replay_mask]
        labels_replay = labels[replay_mask]
        domain_labels_replay = domain_labels[replay_mask]

        #no_grad, unlike requires_grad=False, detaches all elements from the gradient computation graph
        with torch.no_grad():
            base_replay = model.backbone(inputs_replay)
            base_replay = model.pool(base_replay).flatten(1)
            inv_feats_replay = model.invariant(base_replay)

        specific_feats = model.specific(base_replay)
        spec_domain_pred = model.specific_domain_classifier(specific_feats)
        
        if detach_base:
            combined = torch.cat([inv_feats_replay, specific_feats, base_replay], dim=1)
        else:
            combined = torch.cat([inv_feats_replay, specific_feats], dim=1)  
        
        scores = model.head(combined)
        
        task_loss_replay = mse_criterion(scores, labels_replay)
        spec_domain_loss_replay = ce_criterion(spec_domain_pred, domain_labels_replay)
        total_loss_replay = task_loss_replay + 0.2 * spec_domain_loss_replay
        
        total_loss_replay.backward()
    
    metrics = {
        'task_loss': task_loss.item(),
        'inv_domain': inv_domain_loss.item(),
        'spec_domain': spec_domain_loss.item(),
        'similarity': similarity_loss.item(),
        'inv_acc': inv_acc,
        'spec_acc': spec_acc,
        'replay_count': replay_mask.sum().item(),
        'current_count': current_mask.sum().item()
    }
    return total_loss, metrics

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            batch_size = batch[0].shape[0]
            loss, _ = heuristic_dualbranch_batch(model, batch, device, mse_criterion=criterion)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
        val_loss = total_loss/total_samples
    return val_loss

def cross_domain_validation(model, domain_dataloaders, criterion, device, validation_set='val'):
    results = {}
    for domain, loaders in domain_dataloaders.items():
        val_loader = loaders[validation_set]
        val_loss = evaluate_model(model, val_loader, criterion, device)
        results[domain] = val_loss
    return results

def average_metrics(metrics_list):
    # metrics_list: list of dicts, each dict contains metrics for a batch
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    avg_metrics = {}
    for k in keys:
        avg_metrics[k] = float(np.mean([m[k] for m in metrics_list if k in m]))
    return avg_metrics

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

class EarlyStopping:
    def __init__(self, checkpoint_dir: str, model_name: str, patience: int, verbose: bool = True, delta: float = 1e-3):
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name

        self.patience = patience
        self.verbose = verbose
        self.delta = delta

        self.patience_counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss: float, model: torch.nn.Module, optimizer: torch.optim.Optimizer, history: Dict[str, Any], epoch=int) -> bool:
        score = -val_loss  # Since lower val_loss is better, invert for comparison

        if (self.best_score is None) or (score >= self.best_score + self.delta):
            self.best_score = score  
            if self.verbose:
                self.val_loss_min = float(val_loss)
                print(f"Validation loss improved ({self.val_loss_min:.6f} -> {val_loss:.6f}). Saving checkpoint.")
            self._save_checkpoint(model, optimizer, history, epoch)
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.verbose:
                print(f"No improvement. EarlyStopping patience: {self.patience_counter}/{self.patience}")
            if self.patience_counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop

    def _save_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, history: Dict[str, Any], epoch: int) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
        }
        path = os.path.join(self.checkpoint_dir, f"{self.model_name}.pt")
        torch.save(checkpoint, path)

    def restore_best_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, history: Dict[str, Any]) -> int:
        path = os.path.join(self.checkpoint_dir, f"{self.model_name}.pt")
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        history.clear()
        history.update(checkpoint["history"])
        return checkpoint['epoch']


from contextlib import contextmanager
import time

@contextmanager
def timing(history, key):
    """
    To use, add 
    'timings' to history dictionary
    and
    with timing(history['timings'], 'description'):
    around anything to time
    """
    start = time.monotonic()
    yield
    duration = time.monotonic() - start
    history.setdefault(key, []).append(duration)

def unified_train_loop(
    model, domains, domain_dataloaders, buffer, optimizer, device,
    batch_fn, batch_kwargs, num_epochs, exp_name="exp", 
    gradient_clipping=False, collect_tsne_data=False, restart={}, 
    eval_buffer=False, checkpoint_dir="../checkpoints", validation_set='val', scheduler=None, refresh_optimiser=False, early_stopping=True,
):
    scaler = torch.amp.GradScaler('cuda') if torch.device(device).type == "cuda" else None
    if scheduler is not None:
        scheduler, warmup = scheduler
    
    start_domain_idx = 0
    history = {
        'train_epoch_loss': [],
        'val_epoch_loss': [],
        'val_buffer_epoch_loss': [],
        'train_epoch_metrics': [],
        # 'cross_domain_val': [],
        'grad_norms': [],
        # 'timings': {},
    }
    
    if restart:
        # Populate history
        history = restart.get('history', {})
        # Populate buffer
        start_domain_idx = domains.index(restart['domain'])
        for domain_idx, current_domain in enumerate(domains[:start_domain_idx]):
            buffer.update_buffer(current_domain, domain_dataloaders[current_domain]['train'].dataset) 
        print(f"Restarting from domain {restart['domain']} index {start_domain_idx}")
        print(f"Buffer: {buffer.get_domain_distribution()}")         
        

    for domain_idx, current_domain in enumerate(tqdm(domains[start_domain_idx:], desc=f"Total training", disable=TQDM_DISABLED), start=start_domain_idx):
        if TQDM_DISABLED: print(f"[{exp_name}]\t{datetime.datetime.now()}: Starting domain {current_domain}")
        if bool(buffer):
            train_loader = buffer.get_loader_with_replay(current_domain, domain_dataloaders[current_domain]['train'])
        else:
            train_loader = domain_dataloaders[current_domain]['train']
        if eval_buffer:
            eval_loader = eval_buffer.get_loader_with_replay(current_domain, domain_dataloaders[current_domain][validation_set])
            
        len_dataloader = len(train_loader)

        # Initialize new optimiser like the previous one for each domain
        if refresh_optimiser:
            optimizer = type(optimizer)(optimizer.param_groups)

        if scheduler is not None:
            total_training_steps = num_epochs * len_dataloader
            warmup_steps = int(warmup * total_training_steps)
            lr_scheduler = scheduler(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_training_steps
            )
        if early_stopping:
            model_name = exp_name + f"_domain{current_domain}"
            early_stopper = EarlyStopping(checkpoint_dir=checkpoint_dir, model_name=model_name, patience=15, verbose=False, delta=1e-3)

        for epoch in trange(num_epochs, desc=f"Current domain {current_domain}", disable=TQDM_DISABLED):
            if TQDM_DISABLED: print(f"[{exp_name}]\t{datetime.datetime.now()}: Starting epoch {epoch}/{num_epochs}")
            model.train()
            epoch_loss = 0.0
            samples = 0
            batch_metrics_list = []
            
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Current epoch {epoch}", leave=False, disable=TQDM_DISABLED)):
                if not batch_kwargs.get('alpha'):
                    p = (epoch * len_dataloader + batch_idx) / (num_epochs * len_dataloader)
                    alpha = 2. / (1. + np.exp(-10 * p)) - 1
                else:
                    alpha = batch_kwargs['alpha']

                optimizer.zero_grad()

                if torch.device(device).type == "cuda":
                    with torch.autocast('cuda', dtype=torch.float16):
                        loss, metrics = batch_fn(model, batch, device, **{**batch_kwargs, 'current_domain': current_domain, 'alpha':alpha})
                    scaler.scale(loss).backward()
                    if gradient_clipping:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    if scheduler is not None:
                        lr_scheduler.step()
                else:
                    loss, metrics = batch_fn(model, batch, device, **{**batch_kwargs, 'current_domain': current_domain, 'alpha':alpha})
                    loss.backward()
                    if gradient_clipping:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    if scheduler is not None:
                        lr_scheduler.step()
                            
                metrics.setdefault('lrs', []).append(optimizer.param_groups[0]['lr'])

                batch_size = batch[0].size(0)
                epoch_loss += loss.item() * batch_size
                samples += batch_size
                batch_metrics_list.append(metrics)
            
            avg_epoch_loss = epoch_loss / samples
            history['train_epoch_loss'].append(avg_epoch_loss)
            # Average batch metrics for this epoch
            avg_metrics = average_metrics(batch_metrics_list)
            history['train_epoch_metrics'].append(avg_metrics)
            
            # Collect gradients
            grad_norms = collect_gradients(model)
            history['grad_norms'].append(grad_norms)
            
            # Validation on current domain
            val_loss = evaluate_model(model, domain_dataloaders[current_domain][validation_set], batch_kwargs['mse_criterion'], device)    
            history['val_epoch_loss'].append(val_loss)
            if eval_buffer:
                val_loss_buffer = evaluate_model(model, eval_loader, batch_kwargs['mse_criterion'], device)
                history['val_buffer_epoch_loss'].append(val_loss_buffer)
            
            with open(f"{checkpoint_dir}/{exp_name}_history.pkl", "wb") as f:
                pickle.dump(history, f)
            if TQDM_DISABLED: print(f"[{exp_name}]\t{datetime.datetime.now()}: History pickle updated")

            if early_stopping:
                stop = early_stopper(val_loss, model, optimizer, history, epoch)

                if stop or (epoch == num_epochs-1): 
                    best_epoch = early_stopper.restore_best_checkpoint(model, optimizer, history)
                    print(f"Early stopping triggered at domain {current_domain} epoch {epoch}. Model restored to epoch {best_epoch}")
                    break

        # Instead of batchwise average do cross domain validation on inference on all test samples
        # Cross-domain validation (after each domain)
        # cross_val = cross_domain_validation(model, domain_dataloaders, batch_kwargs['mse_criterion'], device=device, validation_set=validation_set)
        # history['cross_domain_val'].append(cross_val)
        
        # Handle saving through EarlyStopper
        # Only save last model per domain to save space
        if not early_stopping:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
            }, f"{checkpoint_dir}/{exp_name}_domain{current_domain}_epoch{epoch}.pt")
            if TQDM_DISABLED: print(f"[{exp_name}]\t{datetime.datetime.now()}: Checkpoint saved at epoch {epoch}")
          
        if bool(buffer):
            buffer.update_buffer(current_domain, domain_dataloaders[current_domain]['train'].dataset)
        if eval_buffer:
            eval_buffer.update_buffer(current_domain, domain_dataloaders[current_domain][validation_set].dataset)
    return history
