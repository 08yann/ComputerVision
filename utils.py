from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
import torchvision
import os
from PIL import Image
from pathlib import Path
from itertools import chain
import numpy as np
import torch
import pickle

def listdir(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext)) for ext in ['png','jpg','jpeg','JPG']]))
    return fnames

# Not necessary since ImageFolder from torchvision already implemented
class AnimalDataset(Dataset):
    def __init__(self, root, transform = None):
        self.samples, self.targets, self.classes = self._make_dataset(root)
        self.transform = transform

    def _make_dataset(self,root):
        classes = os.listdir(root)
        fnames, labels = [],[]
        for idx, domain in enumerate(sorted(classes)):
            class_fnames = listdir(os.path.join(root, domain))
            fnames += class_fnames
            labels += [idx] * len(class_fnames)

        return fnames, labels, classes

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
    classes = data.classes
    return DataLoader(data, batch_size, shuffle = True), classes


def quantisize(images, levels):
    if levels == 2:
        return (torch.bucketize(images, torch.arange(levels).to(images.device) / levels, right = True)-1).float()
    else:
        return (torch.bucketize(images, torch.arange(levels).to(images.device) / levels, right = True)-1).long().squeeze(1)


class MNIST_colorized(Dataset):
    def __init__(self, training = True):
        self.images, self.labels = self._make_dataset(train = training)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.5,0.5,0.5], std = [0.5,0.5,0.5])
        ])

    def _make_dataset(self, train = True):
        path = './MNIST/raw/'
        if train:
            fname = 'train-images-idx3-ubyte'
            label_fname = 'train-labels-idx1-ubyte'
        else:
            fname = 't10k-images-idx3-ubyte'
            label_fname = 't10k-labels-idx1-ubyte'

        with open(path + fname, 'rb') as f:
            dat = f.read()
            images = np.frombuffer(dat,dtype = np.uint8)[0x10:].reshape(-1,28,28)
        
        with open(path + label_fname, 'rb') as f:
            label = f.read()
            labels = np.frombuffer(label,dtype = np.uint8)[8:]
        
        return images, labels

    def __getitem__(self,idx):
        img = self.images[idx]/255.
        lab = self.labels[idx]

        # Colorise using random variable to multiply current black-white picture
        img_color = img[:,:,np.newaxis].repeat(3,axis = -1) * np.clip(np.random.random((img.shape[0], img.shape[1],3)), 0.2,1.)
        img_color = self.transform(img_color)

        return img_color, lab
    
    def __len__(self):
        return len(self.images)



class MnistQuantizedEncoding(Dataset):
    def __init__(self, config, size_quant, fname_encodings = None):
        self.size_codebook = config['model parameters']['size_codebook']
        
        dir_name = './additional_datasets/'
        if fname_encodings is None:
            self.path = dir_name +'mnist_quantized_encodings.pkl'
        else:
            self.path = dir_name + fname_encodings
        self.size_quant = size_quant
        self.quantized = self.load_quantized()

 

    def load_quantized(self):
        mnist_encodings = pickle.load(open(self.path, 'rb'))
        mnist_encodings = mnist_encodings.reshape(mnist_encodings.size(0),self.size_quant, self.size_quant).unsqueeze(1)
        
        return mnist_encodings

    def __len__(self):
        return len(self.quantized)
    
    def __getitem__(self, index):
        context = self.quantized[index]
        return context