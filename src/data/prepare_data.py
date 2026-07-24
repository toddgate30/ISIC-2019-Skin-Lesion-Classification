import torch
import kagglehub
from pathlib import Path
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import random_split, Dataset, DataLoader
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

def prepare_data(config):
    """Loads datasets and return train and validation loaders"""

    # Hard Coding 
    # isic_path = "salviohexia/isic-2019-skin-lesion-images-for-classification"

    dataset_name = config.get("dataset", {}).get("name")
    if dataset_name is None:
        raise KeyError("Missing dataset.name in config file")
    
    raw_dataset_path = config.get("dataset", {}).get("path")
    if raw_dataset_path is None:
        raise KeyError("Missing dataset.path in config file")
    
    dataset_path = Path(kagglehub.dataset_download(raw_dataset_path))

    if dataset_name == "isic2019":
        labels_path = dataset_path / "ISIC_2019_Training_GroundTruth.csv"
    else:
        raise NotImplementedError(f"Please write how to import data from {dataset_name} in src/data/prepare_data.py")

    # Hardcoded mean and std from ImageNet
    inet_mean = [0.485, 0.456, 0.406]
    inet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(inet_mean, inet_std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(inet_mean, inet_std)
    ])
    
    df = pd.read_csv(labels_path)

    label_columns = df.columns.drop["image"]
    stratify_labels = df[label_columns].values.argmax(axis=1)

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True, stratify=stratify_labels)

    train_dataset = ISICDataset(train_df, dataset_path, transform=train_transform)
    val_dataset = ISICDataset(val_df, dataset_path, transform=val_transform)

    metabatch_size = config.get("metabatch_size", 320)
    num_workers = config.get("num_workers", 1)

    train_loader = DataLoader(train_dataset, batch_size=metabatch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=metabatch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
    

prepare_data({"dataset": {"name": "isic2019", "path": "salviohexia/isic-2019-skin-lesion-images-for-classification"}})

class ISICDataset(Dataset):

    def __init__(self, dataframe, image_dir, transform=None):
        self.df = dataframe
        self.image_dir = image_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = self.image_dir / f"{row['image']}.jpg"
        image = Image.open(image_path).convert("RGB")
        label = torch.tensor(row.drop("image").values.argmax(), dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label