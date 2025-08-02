import pandas as pd
from torchvision import transforms
import numpy as np
import random
import torch
import torch.nn as nn
import datetime
import concurrent.futures
import sys, os
import clip

# Add module paths for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.buffers import NaiveRehearsalBuffer
from data_processing.data_processing import get_domain_dataloaders
from models.training_utils import heuristic_dualbranch_batch, unified_train_loop
from models.heuristicSplitModel import DualBranchModel

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # Enforce deterministic algorithms
        torch.backends.cudnn.benchmark = False     # Disable benchmark for reproducibility

    os.environ['PYTHONHASHSEED'] = str(seed)       # Seed Python hashing, which can affect ordering
set_seed(42)


DATA_PATH = "../data/pepper_data_train.pkl"
CHECKPOINT_DIR = "../checkpoints"

domains = ['Home', 'BigOffice-2', 'BigOffice-3', 'Hallway', 'MeetingRoom', 'SmallOffice']
testing_scenarios = {
    'mobilenetv2': (False,),
    'resnet18': (False,),
    'efficientnetb0': (False,),
    'clip': (True,)
}
ablations = ['base', 'no_mask', 'only_soc', 'only_env']
default_transform = transforms.Compose([
    transforms.Resize((144,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def train_scenario(name, freeze_branches, ablation, seed=42):
    # Create a unique log file for each experiment
    log_path = os.path.join(CHECKPOINT_DIR, f"{name}_{ablation}.log")
    logfile = open(log_path, "a", buffering=1)
    sys.stdout = logfile
    sys.stderr = logfile

    try:
        print("="*40)
        print(f"STARTED: {name} | {ablation} | PID: {os.getpid()} | SEED: {seed} | {datetime.datetime.now()}")
        print("="*40)

        if set_seed is not None:
            set_seed(42)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Only load CLIP if needed
        transform = default_transform
        clip_model, clip_transform = (None, None)
        if name == 'clip':
            clip_model, clip_transform = clip.load("ViT-B/32", device=device)
            transform = clip_transform

        # Load dataframe (gcsfuse mount)
        df = pd.read_pickle(DATA_PATH)

        # DataLoader selection by ablation
        if ablation == 'base':
            domain_dataloaders = get_domain_dataloaders(
                df, batch_sizes=(32, 64, 64), double_img=True, transforms=[transform]*2, num_workers=0, include_test=None)
        elif ablation == 'no_mask':
            domain_dataloaders = get_domain_dataloaders(
                df, batch_sizes=(32, 64, 64), double_img=False, transforms=transform, num_workers=0, include_test=None)
        elif ablation == 'only_soc':
            df['image_path'] = df['image_path_social']
            domain_dataloaders = get_domain_dataloaders(
                df, batch_sizes=(32, 64, 64), double_img=False, transforms=transform, num_workers=0, include_test=None)
        elif ablation == 'only_env':
            df['image_path'] = df['image_path_env']
            domain_dataloaders = get_domain_dataloaders(
                df, batch_sizes=(32, 64, 64), double_img=False, transforms=transform, num_workers=0, include_test=None)
        else:
            raise ValueError(f"Unknown ablation: {ablation}")

        print(f"\nTraining: {name} | Ablation: {ablation} | PID: {os.getpid()}")
        setup = {'branch': name} if ablation == 'base' else {'branch': name, 'env': 'ablated'}
        auxilary_model = clip_model if name == 'clip' else None
        model = DualBranchModel(
            dropout_rate=0.1, setup=setup, freeze_branches=freeze_branches, clip_model=auxilary_model
        )
        dual_model = model.to(device)
        trainable_params = [p for p in dual_model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=1e-3)
        buffer = NaiveRehearsalBuffer(buffer_size=120)
        epochs = 30
        exp_name = f"{name}_{ablation}_dropout0.1_epochs{epochs}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        dualbranch_kwargs = {
            'mse_criterion': nn.MSELoss(),
            'ce_criterion': nn.CrossEntropyLoss()
        }
        unified_train_loop(
            model=dual_model,
            domains=domains,
            domain_dataloaders=domain_dataloaders,
            buffer=buffer,
            optimizer=optimizer,
            device=device,
            batch_fn=heuristic_dualbranch_batch,
            batch_kwargs=dualbranch_kwargs,
            num_epochs=epochs,
            exp_name=exp_name,
            gradient_clipping=True,
            collect_tsne_data=False,
            checkpoint_dir=CHECKPOINT_DIR,
            validation_set='val',
        )

        print(f"COMPLETED: {name} | {ablation} | {datetime.datetime.now()}")

    except Exception as exc:
        print(f"FAILED: {name} | {ablation} | {datetime.datetime.now()}")
        import traceback
        traceback.print_exc()
        raise  # propagate to main process, marks job as failed

    finally:
        logfile.close()

if __name__ == "__main__":
    # Build all jobs
    jobs = []
    for name, (freeze_branches,) in testing_scenarios.items():
        for ablation in ablations:
            jobs.append((name, freeze_branches, ablation))

    # Use 4 workers (physical cores)
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(train_scenario, name, fb, ab)
            for name, fb, ab in jobs
        ]
        for f in concurrent.futures.as_completed(futures):
            try:
                f.result()
            except Exception as e:
                print(f"Job failed: {e}")