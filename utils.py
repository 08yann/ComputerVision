from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
import torchvision
import os
from PIL import Image
from pathlib import Path
from itertools import chain
import numpy as np


def listdir(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext)) for ext in ['png','jpg','jpeg','JPG']]))
    return fnames

# Not necessary since ImageFolder from torchvision already implemented
class AnimalDataset(Dataset):
    def __init__(self, root, transform = None):
        self.samples, self.targets = self._make_dataset(root)
        self.transform = transform

    def _make_dataset(self,root):
        classes = os.listdir(root)
        fnames, labels = [],[]
        for idx, domain in enumerate(sorted(classes)):
            class_fnames = listdir(os.path.join(root, domain))
            fnames += class_fnames
            labels += [idx] * len(class_fnames)

        return fnames, labels

    def __getitem__(self,idx):
        fname, label = self.samples[idx], self.targets[idx]
        img = Image.open(fname).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label
    def __len__(self):
        return len(self.targets)


def get_balanced_sampler(labels):
    nb_class = np.bincount(labels)
    class_weight = 1./nb_class
    weights = class_weight[labels]
    return WeightedRandomSampler(weights, len(weights))


def get_dataloader(root = './afhq/', img_size = 224,batch_size = 32, prob = 0.5, split = 'train'):
    if split == 'train' or split == 'val':
        root_folder = root+ split + '/'
        root_folder = root + split+'/'
    else:
        return NotImplementedError
    crop = transforms.RandomResizedCrop(img_size, scale = [0.8,1.0],ratio = [0.9,1.1])
    rand_crop = transforms.Lambda(lambda x: crop(x) if np.random.uniform()<prob else x)

    transform = transforms.Compose([
        rand_crop,
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    data = torchvision.datasets.ImageFolder(root_folder, transform)
    return DataLoader(data, batch_size, shuffle = True)