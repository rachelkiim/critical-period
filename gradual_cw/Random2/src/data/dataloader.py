"""Dataset loading and splitting utilities."""

import torch
import torchvision
import numpy as np

from src.config import ExperimentConfig

# CIFAR-10 normalization constants
CIFAR10_MEAN = [0.49139968, 0.48215841, 0.44653091]
CIFAR10_STD = [0.20220212, 0.19931542, 0.20086346]


def get_cifar10_transform(augmentation=False):
    """Get CIFAR-10 transform pipeline.

    Args:
        augmentation: If True, prepend RandomCrop and RandomHorizontalFlip
                      (for training). Default False (for test/evaluation).
    """
    transforms = []
    if augmentation:
        transforms.append(torchvision.transforms.RandomCrop(32, padding=4))
        transforms.append(torchvision.transforms.RandomHorizontalFlip())
    transforms.append(torchvision.transforms.ToTensor())
    transforms.append(torchvision.transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD))
    return torchvision.transforms.Compose(transforms)


def get_ood_transform():
    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])


def compute_cifar10_mean_std(data_path: str = "./data", batch_size: int = 1024, num_workers: int = 0):
    """Compute CIFAR-10 channel-wise mean/std using per-image averaging.

    This function uses:
        mean_c = mean_i(mean_{h,w}(x_{i,h,w,c}/255))
        std_c  = mean_i(std_{h,w}(x_{i,h,w,c}/255))

    Returns:
        Tuple of ([mean_r, mean_g, mean_b], [std_r, std_g, std_b]).
    """
    dataset = torchvision.datasets.CIFAR10(
        root=data_path,
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    mean_sum = torch.zeros(3, dtype=torch.float64)
    std_sum = torch.zeros(3, dtype=torch.float64)
    num_images = 0

    for images, _ in loader:
        images = images.to(dtype=torch.float64)
        batch_mean = images.mean(dim=(2, 3))
        batch_std = images.std(dim=(2, 3), unbiased=False)
        mean_sum += batch_mean.sum(dim=0)
        std_sum += batch_std.sum(dim=0)
        num_images += images.shape[0]

    mean = mean_sum / num_images
    std = std_sum / num_images

    return mean.tolist(), std.tolist()


def split_dataset(dataset, validation_size=5000, test_size=5000):
    """Split dataset into validation and test subsets.

    Args:
        dataset: The original dataset.
        validation_size: Number of samples for validation set.
        test_size: Number of samples for test set.

    Returns:
        Tuple of (val_dataset, test_dataset) as torch Subsets.
    """
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)

    val_indices = indices[:validation_size]
    test_indices = indices[validation_size:validation_size + test_size]

    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    return val_dataset, test_dataset


def load_datasets(config: ExperimentConfig):
    """Load CIFAR training and test datasets.

    Args:
        config: Experiment configuration containing data_path, num_image, batch_size.

    Returns:
        Tuple of (train_loader, test_loader)
    """
    train_transform = get_cifar10_transform()
    test_transform = get_cifar10_transform()

    # Load CIFAR-10
    train_dataset = torchvision.datasets.CIFAR10(
        root=config.data_path, train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=config.data_path, train=False, download=True, transform=test_transform
    )

    # Subsample training data
    if config.num_image < len(train_dataset):
        indices = np.random.choice(len(train_dataset), config.num_image, replace=False)
        train_dataset = torch.utils.data.Subset(train_dataset, indices)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
    )

    return train_loader, test_loader


def load_ood_datasets(config: ExperimentConfig, max_samples: int = 10000):
    """Load OOD loader from SVHN test split.

    Args:
        config: Experiment configuration containing data_path, batch_size, and seed.
        max_samples: Maximum number of OOD samples to use.

    Returns:
        OOD data loader.
    """
    ood_transform = get_ood_transform()
    ood_dataset = torchvision.datasets.SVHN(
        root=config.data_path, split="test", download=True, transform=ood_transform
    )

    sample_size = min(max_samples, len(ood_dataset))
    rng = np.random.default_rng(config.seed)
    indices = rng.choice(len(ood_dataset), sample_size, replace=False)
    ood_dataset = torch.utils.data.Subset(ood_dataset, indices)

    ood_loader = torch.utils.data.DataLoader(
        ood_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
    )
    return ood_loader
