import os
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms as T
from glob import glob
from PIL import Image
from tqdm import tqdm

def find_dataset(root, split, percent):
    if split=='train':
        # Training set: align + conflict
        print(os.path.join(root, percent, 'align', "*", "*"))
        align = glob(os.path.join(root, percent, 'align', "*", "*"))
        conflict = glob(os.path.join(root, percent, 'conflict', "*", "*"))
        data = align + conflict
    elif split=='test':
        # Test set: align + conflict
        data = glob(os.path.join(root, 'test', "*", "*"))
    elif split=='test-align':
        # Test set: align only
        if root.endswith('cmnist'):
            data_1 = glob(os.path.join(root, percent, 'valid', "*"))
        elif root.endswith('cifar10c'):
            data_1 = glob(os.path.join(root, percent, 'valid', "*", "*"))
        else:
            raise ValueError("Invalid dataset for test-align split")
        
        data_2 = glob(os.path.join(root, 'test', "*", "*"))
        data_2 = [x for x in data_2 if x.split('_')[-2] == x.split('_')[-1].split('.')[0]]
        print(len(data_1), len(data_2))
        data = data_1 + data_2
    elif split=='test-conflict':
        # Test set: conflict only
        data = glob(os.path.join(root, 'test', "*", "*"))    
        data = [x for x in data if x.split('_')[-2] != x.split('_')[-1].split('.')[0]]
    else:
        raise ValueError("Invalid split")

    return data

class CustomDataset(Dataset):
    def __init__(self, 
                 dataset: str,
                 dataset_dir: str,
                 split: str,
                 percent: str,
                 transform: T.Compose = None):
        """
        Initialize the dataset from the given conditions.

        Args:
            dataset (str): Name of the dataset.
                        e.g., 'cifar10c', 'cmnist'
            dataset_dir (str): Directory where the dataset is stored.
            split (str): Split of the dataset.
                                e.g., 'train', 'test', 'test-align', 'test-conflict'
            percent (str): Percent of the dataset.
                        e.g., "0.5pct", "1pct", "2pct", "5pct"
            transform (T.Compose): Transform to apply to the dataset.
        """
        super(CustomDataset, self).__init__()
        self.dataset = dataset
        self.dataset_dir = dataset_dir
        self.split = split
        self.percent = percent
        if transform is None:
            self.transform = T.Compose([T.ToTensor()])
        else:
            self.transform = transform

        self.data = find_dataset(os.path.join(self.dataset_dir, self.dataset), self.split, self.percent)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        attr = torch.LongTensor([
            int(self.data[index].split('_')[-2]),
            int(self.data[index].split('_')[-1].split('.')[0])
            ])
        image = Image.open(self.data[index]).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, attr[0], attr[1]
    
def cache_tensors_to_single_file(data_paths, save_path):
    transform = T.ToTensor()
    cache_data = []

    print(f"Caching {len(data_paths)} samples to {save_path}...")
    for path in tqdm(data_paths):
        image = Image.open(path).convert('RGB')
        tensor = transform(image)
        attr = torch.LongTensor([
            int(path.split('_')[-2]),
            int(path.split('_')[-1].split('.')[0])
        ])
        cache_data.append((tensor, attr[0], attr[1]))

    torch.save(cache_data, save_path)

class CashedCustomDataset(Dataset):
    def __init__(self, dataset, dataset_dir, split, percent, transform=None):
        super().__init__()
        self.transform = transform

        cache_root = os.path.join(dataset_dir, 'cached')
        os.makedirs(cache_root, exist_ok=True)

        # {dataset_dir}/cached/{dataset}_{split}_{percent}.pt
        cache_filename = f"{dataset}_{split}_{percent}.pt"
        cache_path = os.path.join(cache_root, cache_filename)

        if not os.path.exists(cache_path):
            print(f"Cached file not found at {cache_path}. Generating cache...")
            data_paths = find_dataset(os.path.join(dataset_dir, dataset), split, percent)
            cache_tensors_to_single_file(data_paths, cache_path)
        else:
            print(f"Loading cached tensor file from {cache_path}")

        self.data = torch.load(cache_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, attr0, attr1 = self.data[index]
        if self.transform:
            image = self.transform(image)
        return image, attr0, attr1



if __name__ == '__main__':
    dataset = "cmnist"
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../dataset'))
    split = "test-align"
    percent = "0.5pct"
 
    data = find_dataset(os.path.join(dataset_dir, dataset), split, percent)
    print("Number of data:", len(data))
    print("Examples:", data[:10])
    