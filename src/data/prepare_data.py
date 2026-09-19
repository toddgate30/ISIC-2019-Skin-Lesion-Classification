import torch
import kagglehub
from pathlib import Path
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import random_split, Dataset, DataLoader
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import random
import numpy as np

class ISICDataset(Dataset):
    """PyTorch dataset for loading ISIC images and their class labels.

    Each row of the dataframe corresponds to an image and contains one-hot
    encoded class labels. The dataset uses the image name and its class label
    to locate the corresponding image file in the directory structure.

    Args:
        dataframe: Dataframe containing image names in the ``"image"`` column
            and one-hot encoded class labels in the remaining columns.
        image_dir: Root directory containing one subdirectory for each class.
        transform: Optional torchvision transform applied to each image after
            it is loaded.
    """
    def __init__(self, dataframe, image_dir, transform=None):
        self.df = dataframe
        self.image_dir = image_dir
        self.transform = transform
        self.label_columns = dataframe.columns.drop("image")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """Load an image and its class label.

        The class label is determined from the one-hot encoded label columns
        in the corresponding dataframe row. The image is loaded from the
        subdirectory associated with that class and converted to RGB.

        Args:
            idx: Index of the image to retrieve.

        Returns:
            A tuple containing the transformed image and its class label as a
            ``torch.long`` tensor.
        """
        row = self.df.iloc[idx]

        label = row[self.label_columns].values.argmax()
        class_name = self.label_columns[label]
        image_path = self.image_dir / class_name / f"{row['image']}.jpg"

        image = Image.open(image_path).convert("RGB")
        label = torch.tensor(row.drop("image").values.argmax(), dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label

def prepare_data(config):
    """Prepare the dataset, transforms, and data loaders for training.

    Loads the dataset specified in the configuration, creates training and
    validation splits using stratified sampling, and constructs separate
    transforms for training and validation. The training transform includes
    random data augmentation, while the validation transform only performs
    resizing and normalization.

    The random number generators for PyTorch, Python, NumPy, and CUDA are
    seeded using the project seed from the configuration to make the data
    split and training setup reproducible.

    Args:
        config: Configuration dictionary containing the dataset settings,
            project seed, and data loader parameters.

    Returns:
        A tuple containing:
            train_loader: Data loader that provides shuffled training
                meta-batches.
            train_metrics_loader: Data loader that provides the full training
                dataset in fixed batches for computing training metrics.
            val_loader: Data loader that provides the validation dataset in
                fixed batches.
            class_counts: Number of training examples belonging to each class.
            class_names: Names of the dataset classes in their corresponding
                label-index order.

    Raises:
        KeyError: If ``dataset.name`` or ``dataset.path`` is missing from the
            configuration.
        NotImplementedError: If the configured dataset is not supported.
    """
    # Validates the dataset name and path
    dataset_name = config.get("dataset", {}).get("name")
    if dataset_name is None:
        raise KeyError("Missing dataset.name in config file")
    
    raw_dataset_path = config.get("dataset", {}).get("path")
    if raw_dataset_path is None:
        raise KeyError("Missing dataset.path in config file")
    
    # dataset_path = Path(kagglehub.dataset_download(raw_dataset_path))
    dataset_path = Path("~/.cache/kagglehub/datasets/salviohexia/isic-2019-skin-lesion-images-for-classification/versions/1/").expanduser().resolve()

    if dataset_name == "isic2019":
        labels_path = dataset_path / "ISIC_2019_Training_GroundTruth.csv"
    # When working with other dataset, place an elif statement here.
    else:
        raise NotImplementedError(f"Please write how to import data from {dataset_name} in src/data/prepare_data.py")

    # Hardcoded mean and std from ImageNet
    inet_mean = [0.485, 0.456, 0.406]
    inet_std = [0.229, 0.224, 0.225]

    # Creates transforms
    rotation = config.get("data_params", {}).get("transform_rotation", 20)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(rotation),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05
        ),
        transforms.ToTensor(),
        transforms.Normalize(inet_mean, inet_std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(inet_mean, inet_std)
    ])

    # Pulls the labels and splits the data
    df = pd.read_csv(labels_path)

    label_columns = df.columns.drop("image")
    stratify_labels = df[label_columns].values.argmax(axis=1)
    class_names = label_columns.tolist()

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True, stratify=stratify_labels)

    # Sets RNG
    seed = config["project"]["seed"]
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Creates datasets
    train_dataset = ISICDataset(train_df, dataset_path, transform=train_transform)
    val_dataset = ISICDataset(val_df, dataset_path, transform=val_transform)

    metabatch_size = config.get("metabatch_size", 320)
    num_workers = config.get("num_workers", 1)

    # Creates loaders
    train_loader = DataLoader(train_dataset, batch_size=metabatch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    train_metrics_loader = DataLoader(train_dataset, batch_size=298, shuffle=False, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=298, shuffle=False, num_workers=num_workers)

    train_labels = train_df[label_columns].values.argmax(axis=1)
    class_counts = np.bincount(train_labels)

    return train_loader, train_metrics_loader, val_loader, class_counts, class_names
    

# prepare_data({"dataset": {"name": "isic2019", "path": "salviohexia/isic-2019-skin-lesion-images-for-classification"}})