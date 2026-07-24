import torch
import kagglehub
from pathlib import Path
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import random_split, Dataset
import pandas as pd
from PIL import Image

def prepare_data(config):
    """Loads datasets and return train, validation, and test loaders"""

    # Hard Coding 
    isic_path = "salviohexia/isic-2019-skin-lesion-images-for-classification"

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
    
    df = pd.read_csv(labels_path)

    
    # raise NotImplementedError

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
        label = row.drop("image").values.astype("float32")

        if self.transform:
            image = self.transform(image)

        return image, label