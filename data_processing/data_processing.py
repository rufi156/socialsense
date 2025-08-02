import argparse
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import random
SEED = 42
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # Enforce deterministic algorithms
        torch.backends.cudnn.benchmark = False     # Disable benchmark for reproducibility

    os.environ['PYTHONHASHSEED'] = str(seed)       # Seed Python hashing, which can affect ordering
set_seed(SEED)

DATASET_DIR = (Path("..") / ".." / "datasets").resolve()
DATASETS = ["OFFICE-MANNERSDB", "MANNERSDBPlus"]
LABEL_COLS = [
    "Vaccum Cleaning", "Mopping the Floor", "Carry Warm Food",
    "Carry Cold Food", "Carry Drinks", "Carry Small Objects",
    "Carry Large Objects", "Cleaning", "Starting a conversation"
]
# mean=[0.485, 0.456, 0.406, 2.142]
# std=[0.229, 0.224, 0.225, 0.931]

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
NORM_VALUES=(mean,std)

def process_csv(csv_path, dataset):
    """Process individual CSV files"""
    df = pd.read_csv(csv_path)
    df = df.drop(columns=df.columns[-1])
    
    # Extract metadata from first column
    first_col = df.columns[0]
    split_data = df[first_col].str.split('_', n=2, expand=True)
    
    df["robot"] = split_data[0]
    df["domain"] = split_data[1]
    df["image_ref"] = split_data[2].astype(int)
    df["dataset"] = dataset

    df = df.drop(columns=[first_col])
    
    return df

def consolidate_data(datasets):
    """Aggregate all CSVs"""
    all_dfs = []
    for dataset in datasets:
        source_path = DATASET_DIR / dataset
        
        for robot in ["NAO", "Pepper", "PR2"]:
            ann_dir = source_path / robot / "Annotations"
            if not ann_dir.exists():
                raise ValueError(f"Labels csv file path ({ann_dir}) doesn't exist")
                
            
            for csv_file in ann_dir.glob("*.csv"):
                try:
                    df = process_csv(csv_file, dataset)
                    all_dfs.append(df)
                except Exception as e:
                    print(f"Error processing {csv_file}: {str(e)}")
    
    df = pd.concat(all_dfs, ignore_index=True)

    return df

def validate_raw_data(df):
    """Comprehensive data quality checks for raw annotation data"""
    required_columns = {'robot', 'domain', 'image_ref', 'dataset'}

    # Check for any missing columns
    missing_cols = required_columns - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Label value validation (should be between 1 and 5)
    for col in LABEL_COLS:
        if df[col].min() < 1 or df[col].max() > 5:
            raise ValueError(f"Label {col} has invalid range [{df[col].min()}, {df[col].max()}]")

    # Null values check
    null_cols = df.columns[df.isnull().any()].tolist()
    if null_cols:
        raise ValueError(f"Null values found in columns: {null_cols}")

    # Data type and value validation for image_ref
    if not pd.api.types.is_integer_dtype(df['image_ref']):
        raise TypeError("image_ref must be integer type")
    if (df['image_ref'] < 0).any():
        raise ValueError("image_ref contains negative values, which is invalid")

    # Categorical value validation
    valid_robots = {'NAO', 'Pepper', 'PR2'}
    invalid_robots = set(df['robot']) - valid_robots
    if invalid_robots:
        raise ValueError(f"Invalid robot values: {invalid_robots}")

    valid_sources = {'OFFICE-MANNERSDB', 'MANNERSDBPlus'}
    invalid_sources = set(df['dataset']) - valid_sources
    if invalid_sources:
        raise ValueError(f"Invalid source directories: {invalid_sources}")

    return True

def aggregate_labels(df):
    """Aggregate multiple annotations per image by image path"""    
    agg_dict = {
        **{col: 'mean' for col in LABEL_COLS},
        **{col: 'first' for col in df.columns.difference(LABEL_COLS).tolist()},
    }
    
    return df.groupby('image_path', as_index=False).agg(agg_dict)

def resolve_image_path(row):
    """Robust path resolution with validation"""
    base_dir = DATASET_DIR / row['dataset'] / row['robot'] / "Images"
    
    if row['dataset'] == "OFFICE-MANNERSDB":
        target = base_dir / f"{row['domain']}_{row['image_ref']}.png"
    else:
        target = next(base_dir.glob(f"{row['image_ref']}_*.png"), None)
    
    if target and target.exists():
        return str(target.resolve())
    return None

def validate_final_data(df):
    """Final validation after aggregation"""
    # Missing image paths
    missing = df[df['image_path'].isnull()]
    if not missing.empty:
        raise FileNotFoundError(
            f"{len(missing)} images missing after aggregation. Examples:\n"
            f"{missing[['robot', 'domain', 'image_ref']].head()}"
        )
    
    # Null values check
    null_cols = df.columns[df.isnull().any()].tolist()
    if null_cols:
        raise ValueError(f"Null values found in columns: {null_cols}")

    # Duplicate image paths
    duplicates = df[df.duplicated('image_path', keep=False)]
    if not duplicates.empty:
        raise RuntimeError(
            f"Duplicate image paths after aggregation:\n"
            f"{duplicates['image_path'].unique()}"
        )

    # Label validity (1-5)
    for col in LABEL_COLS:
        if df[col].min() < 1 or df[col].max() > 5:
            raise ValueError(
                f"Aggregated label {col} out of range: "
                f"[{df[col].min()}, {df[col].max()}]"
            )

    return True

import torchvision.transforms.functional as F

class DualImageDataset(Dataset):
    def __init__(self, df, depth_transform=None, yolo_transform=None, resize_img_to=(224,224)):
        self.df = df.reset_index(drop=True)
        self.resize_img_to = resize_img_to
        self.depth_transform = depth_transform or self._get_depth_transform()
        self.yolo_transform = yolo_transform or self._get_yolo_transform()
        self.domain_to_index = {domain: idx for idx, domain in enumerate(df['domain'].unique())}

    
    def _get_depth_transform(self,):
        return transforms.Compose([
            transforms.Resize(self.resize_img_to),
            transforms.ToTensor(),
            transforms.Normalize(*NORM_VALUES)
        ])
    
    def _get_yolo_transform(self):
        return transforms.Compose([
            transforms.Resize(self.resize_img_to),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
            
        
        if 'image_path_social' in self.df.columns:
            img_path_env = str(self.df.iloc[idx]["image_path_env"])
            img_path_social = str(self.df.iloc[idx]["image_path_social"])
            try:
                image_env = Image.open(img_path_env).convert("RGB")
            except Exception as e:
                raise RuntimeError(f"Error loading {img_path_env}: {str(e)}")
            try:
                image_social = Image.open(img_path_social).convert("RGB")
            except Exception as e:
                raise RuntimeError(f"Error loading {img_path_social}: {str(e)}")
            
            depth_image = self.depth_transform(image_social)
            yolo_image = self.yolo_transform(image_env)

        else:
            img_path = str(self.df.iloc[idx]["image_path"])
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                raise RuntimeError(f"Error loading {img_path}: {str(e)}")
        
            # Create separate inputs for each backbone
            depth_image = self.depth_transform(image)
            yolo_image = self.yolo_transform(image)
        
        raw_labels = self.df.iloc[idx][LABEL_COLS].values.astype(np.float32)
        scaled_labels = (raw_labels - 1) / 4
        
        domain_labels = self.df.iloc[idx]['domain']
        domain_index = self.domain_to_index[domain_labels]
        
        return depth_image, yolo_image, torch.from_numpy(scaled_labels), domain_index

class ImageLabelDataset(Dataset):
    def __init__(self, df, transform=None, resize_img_to=(288, 512), return_labels=True):
        self.df = df.reset_index(drop=True)
        self.resize_img_to = resize_img_to
        self.transform = transform or self._get_transform()
        self.return_labels = return_labels
        self.domain_to_index = {domain: idx for idx, domain in enumerate(df['domain'].unique())}
        
    
    def _get_transform(self):
        return transforms.Compose([
            transforms.Resize(self.resize_img_to),
            transforms.ToTensor(),
            transforms.Normalize(*NORM_VALUES)
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
            
        img_path = str(self.df.iloc[idx]["image_path"])
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error loading {img_path}: {str(e)}")
        
        image = self.transform(image)

        raw_labels = self.df.iloc[idx][LABEL_COLS].values.astype(np.float32)
        scaled_labels = (raw_labels - 1) / 4
        
        domain_labels = self.df.iloc[idx]['domain']
        domain_index = self.domain_to_index[domain_labels]
  
        return (image, torch.from_numpy(scaled_labels), domain_index) if self.return_labels else image

def _create_dataloaders(df, batch_sizes=(32, 64, 64), resize_img_to=(288, 512), seed=42, double_img=False, transforms=None, num_workers=0, include_test=None):
    """Create train/val dataloaders using image_path as unique key"""
    
    # Get image paths as indexing for split
    unique_images = df[['image_path']].reset_index(drop=True)
    
    # Split using image_path as key
    train_paths, val_paths = train_test_split(
        unique_images['image_path'], 
        test_size=0.25, 
        random_state=seed
    )
   
    # Create subsets
    train_df = df[df['image_path'].isin(train_paths)].reset_index(drop=True)
    val_df = df[df['image_path'].isin(val_paths)].reset_index(drop=True)
    

    # Create datasets
    if double_img:
        train_dataset = DualImageDataset(train_df, transforms[0], transforms[1], resize_img_to=resize_img_to)
        val_dataset = DualImageDataset(val_df, transforms[0], transforms[1], resize_img_to=resize_img_to)
    else:
        train_dataset = ImageLabelDataset(train_df, transform=transforms, resize_img_to=resize_img_to)
        val_dataset = ImageLabelDataset(val_df, transform=transforms, resize_img_to=resize_img_to)
    
    # Create loaders
    num_workers = 0
    loaders = {
        'train': DataLoader(train_dataset, batch_size=batch_sizes[0], shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available()),
        'val': DataLoader(val_dataset, batch_size=batch_sizes[1], shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    }
    
    if include_test is not None:
        test_df = include_test.reset_index(drop=True)
        if double_img:
            test_dataset = DualImageDataset(test_df, transforms[0], transforms[1], resize_img_to=resize_img_to)
        else:
            test_dataset = ImageLabelDataset(test_df, transform=transforms, resize_img_to=resize_img_to)
        loaders['test']= DataLoader(test_dataset, batch_size=batch_sizes[2], shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())

    split_idx = {'train': train_paths, 'val': val_paths}

    return (loaders, split_idx)


def get_domain_dataloaders(df, return_splits=False, batch_sizes=(32, 64, 64), resize_img_to=(288, 512), seed=42, double_img=False, transforms=None, num_workers=0, include_test=None):
    """
    Creates domain stratifed dataloaders

    resize notes:
    for yolo_depthanything model 224,224
    otherwise 512, 288
    LGR had 128,128 
    MobileNetv2 had 224, 224
    """
    domain_dataloaders = {}
    domain_splits = {}
    for domain in df['domain'].unique():
        domain_df = df[df['domain'] == domain]
        loaders, split_idx = _create_dataloaders(domain_df, batch_sizes=batch_sizes, resize_img_to=resize_img_to, seed=seed, double_img=double_img, transforms=transforms, num_workers=num_workers, include_test=include_test)
        domain_dataloaders[domain] = loaders
        domain_splits[domain] = split_idx

    return domain_dataloaders if not return_splits else (domain_dataloaders, domain_splits)


def create_crossvalidation_loaders(df, folds=5, batch_sizes=(32, 64, 64), resize_img_to=(128, 128), num_workers=0):

    transform = transforms.Compose([
            transforms.Resize(resize_img_to),
            transforms.ToTensor(),
            transforms.Normalize(*NORM_VALUES)
        ])

    fold_loaders = {}

    # Ignore negative number folds - treat them as exceptions
    folds = [i for i in range(folds)]
    df = df[df['fold'].isin(folds)]
    
    for fold_num in folds:
        train_df = df[df['fold'] != fold_num].reset_index(drop=True)
        val_df = df[df['fold'] == fold_num].reset_index(drop=True)

        domains = df['domain'].unique()
        
        domain_loaders = {}
        for domain in domains:
            domain_train = train_df[train_df['domain'] == domain].reset_index(drop=True)
            domain_val = val_df[val_df['domain'] == domain].reset_index(drop=True)
            
            train_dataset = ImageLabelDataset(domain_train, transform)
            val_dataset = ImageLabelDataset(domain_val, transform)
        
            domain_loaders[domain] = {
                'train': DataLoader(train_dataset, batch_size=batch_sizes[0], shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available()),
                'val': DataLoader(val_dataset, batch_size=batch_sizes[1], shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
            }
        fold_loaders[fold_num] = domain_loaders
        
    return fold_loaders

import shutil

def extract_data_subset(df, robot_name='Pepper'):
    # Modify the saving section at the end
    # Create the required directory structure
    project_root = Path.cwd().parent  # Goes up from experiments/ to project/
    data_dir = project_root / "data"
    images_dir = data_dir / "images"

    images_dir.mkdir(parents=True, exist_ok=True)

    # Filter for Pepper first
    pepper_df = df[df['robot'] == robot_name].copy()
    pepper_df = pepper_df.reset_index(drop=True)

    # Copy images and update paths
    def copy_and_update_path(row):
        src_path = Path(row['image_path'])
        dst_path = images_dir / src_path.name
        
        if not dst_path.exists():
            shutil.copy2(src_path, dst_path)
        
        return str(dst_path.relative_to(project_root))

    pepper_df['image_path'] = pepper_df.apply(copy_and_update_path, axis=1)

    # Save the filtered dataframe
    pepper_df.to_pickle(data_dir / "pepper_data.pkl")
    return pepper_df

def main(aggregate_by='robot'):
    try:
        raw_df = consolidate_data(DATASETS)
        validate_raw_data(raw_df)
        raw_df['image_path'] = raw_df.apply(resolve_image_path, axis=1)
        if aggregate_by:
            aggregated_df = aggregate_labels(raw_df)
            validate_final_data(aggregated_df)
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        raise

    if aggregate_by:
        aggregated_df.to_pickle("../data/processed_all_data.pkl")
    else:
        raw_df.to_pickle("../data/raw_all_data.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate_by", nargs="?", const="robot", default="robot",
                        help='Set to "robot" (default) or "none" to skip aggregation')
    args = parser.parse_args()

    arg = None if args.aggregate_by.lower() == "none" else args.aggregate_by
    main(aggregate_by=arg)


# ============= LEGACY =================
# ----------- Legacy code for splitting dataset, generating dataloaders for train, validation and test. ------------
# Since then we switched to permanent separation of train vs test and changing all training datast operations to contain only train and val datasets.

def legacy_create_dataloaders(df, batch_sizes=(32, 64, 64), resize_img_to=(288, 512), seed=42, return_splits=False, double_img=False, transforms=None, num_workers=0):
    """Create train/val/test dataloaders using image_path as unique key"""
    
    # Get image paths as indexing for split
    unique_images = df[['image_path']].reset_index(drop=True)
    
    # Split using image_path as key
    train_paths, temp_paths = train_test_split(
        unique_images['image_path'], 
        test_size=0.4, 
        random_state=seed
    )
    val_paths, test_paths = train_test_split(
        temp_paths,
        test_size=0.5, 
        random_state=seed
    )
   
    # Create subsets
    train_df = df[df['image_path'].isin(train_paths)].reset_index(drop=True)
    val_df = df[df['image_path'].isin(val_paths)].reset_index(drop=True)
    test_df = df[df['image_path'].isin(test_paths)].reset_index(drop=True)
    

    # Create datasets
    if double_img:
        train_dataset = DualImageDataset(train_df, transforms[0], transforms[1], resize_img_to=resize_img_to)
        val_dataset = DualImageDataset(val_df, transforms[0], transforms[1], resize_img_to=resize_img_to)
        test_dataset = DualImageDataset(test_df, transforms[0], transforms[1], resize_img_to=resize_img_to)
    else:
        train_dataset = ImageLabelDataset(train_df, transform=transforms, resize_img_to=resize_img_to)
        val_dataset = ImageLabelDataset(val_df, transform=transforms, resize_img_to=resize_img_to)
        test_dataset = ImageLabelDataset(test_df, transform=transforms, resize_img_to=resize_img_to)
    
    # Create loaders
    num_workers = 0
    loaders = {
        'train': DataLoader(train_dataset, batch_size=batch_sizes[0], shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available()),
        'val': DataLoader(val_dataset, batch_size=batch_sizes[1], shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available()),
        'test': DataLoader(test_dataset, batch_size=batch_sizes[2], shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    }
    
    if return_splits:
        split_idx = {'train': train_paths, 'val': val_paths, 'test': test_paths}
        return loaders, split_idx
    else:
        return loaders

def legacy_get_dataloader(df, batch_sizes=(32, 64, 64), resize_img_to=(224,224), return_splits=False, double_img=False, transforms=None, num_workers=0):
    """
    Creates domain stratifed dataloaders
    Returns:
    dataloaders and a cumulative indexes for the test set for checking replicability

    resize notes:
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
        loaders, split_idx = legacy_create_dataloaders(domain_df, batch_sizes=batch_sizes, resize_img_to=resize_img_to, seed=SEED, return_splits=True, double_img=double_img, transforms=transforms, num_workers=num_workers)
        domain_dataloaders[domain] = loaders
        test_split_idx.update(set(split_idx['test']))

    return domain_dataloaders if not return_splits else (domain_dataloaders, test_split_idx)

#----------- One off separation of test set for replicability.------------

def _separate_train_test():
    source_dir="../data/pepper_data.pkl"
    transform = transforms.Compose([
        transforms.Resize((144,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])
    df = pd.read_pickle(source_dir)
    df['image_path_env'] = df['image_path'].apply(lambda p: str(Path('../data/masked/environment') / Path(p).name))
    df['image_path_social'] = df['image_path'].apply(lambda p: str(Path('../data/masked/social') / Path(p).name))
    _, splits1 = legacy_get_dataloader(df, batch_sizes=(32, 64, 64), return_splits=True, double_img=False, transforms=transform, num_workers=0)
    # Normalise paths
    normalized_set = {Path(p).as_posix() for p in splits1}
    df['image_path'] = df['image_path'].apply(lambda x: Path(x).as_posix())
    df['image_path_env'] = df['image_path_env'].apply(lambda x: Path(x).as_posix())
    df['image_path_social'] = df['image_path_social'].apply(lambda x: Path(x).as_posix())

    df_test = df[df['image_path'].isin(normalized_set)]
    df_train = df[~df['image_path'].isin(normalized_set)]

    assert df_train['image_path'].isin(df_test['image_path']).any() == False
    assert df_test['image_path'].isin(df_train['image_path']).any() == False

    df_test.to_pickle(source_dir.replace('.pkl', '_test.pkl'))
    df_train.to_pickle(source_dir.replace('.pkl', '_train.pkl'))