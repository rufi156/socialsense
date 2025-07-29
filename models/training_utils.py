import torch
import torch.nn as nn
from tqdm.notebook import tqdm, trange
import numpy as np
import pickle

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
    inputs1, inputs2, labels, domain_labels = batch
    inputs1, inputs2, labels, domain_labels = inputs1.to(device), inputs2.to(device), labels.to(device), domain_labels.to(device)
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

    return (val_loss, tsne) if tsne else val_loss

def cross_domain_validation(model, domain_dataloaders, criterion, device, validation_set='val', tsne=None):
    results = {}
    for domain, loaders in domain_dataloaders.items():
        val_loader = loaders[validation_set]
        if tsne:
            val_loss, tsne = evaluate_model(model, val_loader, criterion, device, tsne)
        else:
            val_loss = evaluate_model(model, val_loader, criterion, device)
        results[domain] = val_loss
    return (results, tsne) if tsne else results

def average_metrics(metrics_list):
    # metrics_list: list of dicts, each dict contains metrics for a batch
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    avg_metrics = {}
    for k in keys:
        avg_metrics[k] = float(np.mean([m[k] for m in metrics_list if k in m]))
    return avg_metrics

def collect_tsne_features(model, domain_dataloaders, device):
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


def unified_train_loop(
    model, domains, domain_dataloaders, buffer, optimizer, device,
    batch_fn, batch_kwargs, num_epochs=5, exp_name="exp", 
    gradient_clipping=False, collect_tsne_data=False, restart={}, 
    eval_buffer=False, checkpoint_dir="../checkpoints", validation_set='val'
):
# detach_base, binary, full_replay, loss_params={} moved to kwargs
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
            eval_loader = eval_buffer.get_loader_with_replay(current_domain, domain_dataloaders[current_domain][validation_set])
            
        len_dataloader = len(train_loader)
        
        for epoch in trange(num_epochs, desc=f"Current domain {current_domain}"):
            model.train()
            epoch_loss = 0.0
            samples = 0
            batch_metrics_list = []
            
            # for batch_idx, batch in enumerate(train_loader):
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Current epoch {epoch}", leave=False)):
                if not batch_kwargs.get('alpha'):
                    p = (epoch * len_dataloader + batch_idx) / (num_epochs * len_dataloader)
                    alpha = 2. / (1. + np.exp(-10 * p)) - 1
                else:
                    alpha = batch_kwargs['alpha']

                optimizer.zero_grad()
                loss, metrics = batch_fn(model, batch, device, **{**batch_kwargs, 'current_domain': current_domain, 'alpha':alpha})
                if gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                batch_size = batch[0].size(0)
                epoch_loss += loss.item() * batch_size
                samples += batch_size
                global_step += 1
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
                    cross_val, tsne_data = cross_domain_validation(model, domain_dataloaders, batch_kwargs['mse_criterion'], device=device, validation_set=validation_set, tsne=tsne)
                else:
                    cross_val = cross_domain_validation(model, domain_dataloaders, batch_kwargs['mse_criterion'], device=device, validation_set=validation_set)
                history['cross_domain_val'].append(cross_val)

                # Only save last model per domain to save space
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'history': history,
                    'tsne' : tsne_data,
                }, f"{checkpoint_dir}/{exp_name}_domain{current_domain}_epoch{epoch}_step{global_step}.pt")
            # else:
            #     # Save metrics
            #     torch.save({
            #         # 'model_state_dict': model.state_dict(),
            #         # 'optimizer_state_dict': optimizer.state_dict(),
            #         'history': history,
            #         'tsne' : tsne_data,
            #     }, f"../checkpoints/{exp_name}_domain{current_domain}_epoch{epoch}_step{global_step}.pt")
            with open(f"{checkpoint_dir}/{exp_name}_history.pkl", "wb") as f:
                pickle.dump(history, f)
            
        buffer.update_buffer(current_domain, domain_dataloaders[current_domain]['train'].dataset)
        if eval_buffer:
            eval_buffer.update_buffer(current_domain, domain_dataloaders[current_domain][validation_set].dataset)
    return history

