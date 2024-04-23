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
import tqdm
import matplotlib.pyplot as plt

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


def get_dataloader(root = './afhq/', img_size = 224,batch_size = 32, prob = 0.5, split = 'train', animals = None, norm_values = None):
    if split == 'train' or split == 'val':
        root_folder = root+ split + '/'
        root_folder = root + split+'/'
    else:
        return NotImplementedError

    crop = transforms.RandomResizedCrop(img_size, scale = [0.8,1.0],ratio = [0.9,1.1])
    rand_crop = transforms.Lambda(lambda x: crop(x) if np.random.uniform()<prob else x)

    if norm_values is not None:
        assert len(norm_values) == 2

        mean = norm_values[0]
        std = norm_values[1]
    else:
        mean = (0.5,0.5,0.5)
        std = (0.5,0.5,0.5)
    transform = transforms.Compose([
        #rand_crop,
        transforms.Resize((img_size,img_size)),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
        ])
    data = torchvision.datasets.ImageFolder(root_folder, transform)
    classes = data.classes
    if animals is not None:
        idx_keep = [i for i in range(len(data)) if data.imgs[i][1] in [data.class_to_idx[a] for a in animals]]
        classes = animals

        data = torch.utils.data.Subset(data, idx_keep)
    
    unnormalizer = transforms.Compose([
        transforms.Normalize(mean = 0., std = 1/torch.tensor(std)),
        transforms.Normalize(mean = -torch.tensor(mean), std = 1.)
    ])

    return DataLoader(data, batch_size, shuffle = True), unnormalizer, classes


def quantisize(images, levels):
    if levels == 2:
        return (torch.bucketize(images, torch.arange(levels).to(images.device) / levels, right = True)-1).float()
    else:
        return (torch.bucketize(images, torch.arange(levels).to(images.device) / levels, right = True)-1).long().squeeze(1)


class MNIST_colorized(Dataset):
    def __init__(self, training = True, img_size=28, mean =[0.5,0.5,0.5], std = [0.5,0.5,0.5] ):
        self.images, self.labels = self._make_dataset(train = training)
        self.class_to_idx = {str(idx):idx for idx in np.unique(self.labels)}

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(img_size),
            transforms.Normalize(mean = mean, std = std )
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

        return img_color.float(), lab
    
    def __len__(self):
        return len(self.images)



class MnistQuantizedEncoding(Dataset):
    def __init__(self, size_codebook, size_quant, fname_encodings = None):
        self.size_codebook = size_codebook
        
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


class Annealizer:
    def __init__(self,total_steps, cyclical = False):
        self.total_steps = total_steps
        self.current_step = 0

        self.cyclical = cyclical

    
    def __call__(self, loss_kl):
        return self.beta() * loss_kl

    def beta(self):
        coeff = self.current_step/self.total_steps
        return coeff

    
    def step(self):
        if self.current_step< self.total_steps:
            self.current_step += 1
        
        if self.cyclical & self.current_step == self.total_steps:
            self.current_step = 0


def load_dataset(config, training = True):
    assert config['dataset']['name'] in ('MNIST','colorMNIST', 'AFHQ', 'OXFORD', 'CelebA')
    try:
        mean, std = config['dataset']['mean'], config['dataset']['std']
    except: 
        mean, std = 0., 1.
        
    # Since the inputs are normalized, for nicer outputs, we want to reverse this transformation
    unnormalizer = transforms.Compose([transforms.Normalize(mean = 0, std = 1/torch.tensor(std)),transforms.Normalize(mean = -torch.tensor(mean), std = 1.) ])

    if config['dataset']['name'] == 'MNIST':
        dataset = torchvision.datasets.MNIST(root = './data/',train = training, download = True, transform=torchvision.transforms.Compose([transforms.Resize((config['dataset']['img_size'],config['dataset']['img_size'])),torchvision.transforms.ToTensor(), transforms.Normalize(mean = mean, std = std)]))
    elif config['dataset']['name'] == 'colorMNIST':
        dataset = MNIST_colorized(training=training, img_size=config['dataset']['img_size'], mean = mean, std = std)
    elif config['dataset']['name'] == 'OXFORD':
        split = 'trainval' if training else 'test'
        dataset = torchvision.datasets.OxfordIIITPet(root = './data/', download = True, split = split, transform=torchvision.transforms.Compose([transforms.Resize((config['dataset']['img_size'],config['dataset']['img_size'])),torchvision.transforms.ToTensor(),transforms.Normalize(mean = mean, std = std)]))
    elif config['dataset']['name'] == 'CelebA':
        split = 'train' if training else 'test'
        dataset = torchvision.datasets.CelebA(root = './data/', download = True, split = split, target_type='identity', transform=torchvision.transforms.Compose([transforms.Resize((config['dataset']['img_size'],config['dataset']['img_size'])),torchvision.transforms.ToTensor(),transforms.Normalize(mean = mean, std = std)]))
    else:
        root = './afhq/'
        root += 'train' if training else 'val'
        crop = transforms.RandomResizedCrop(config['dataset']['img_size'], scale = [0.8,1.0],ratio = [0.9,1.1])
        prob = 0.5
        rand_crop = transforms.Lambda(lambda x: crop(x) if np.random.uniform()<prob else x)
        transform = transforms.Compose([rand_crop,transforms.Resize((config['dataset']['img_size'],config['dataset']['img_size'])),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(mean=mean, std=std)])
        dataset = torchvision.datasets.ImageFolder(root, transform)
        classes = dataset.classes
        if 'animals' in config['dataset']:
            idx_keep = [i for i in range(len(data)) if dataset.imgs[i][1] in [dataset.class_to_idx[a] for a in config['dataset']['animals']]]
            dataset = torch.utils.dataset.Subset(data, idx_keep)
    
    nb_classes = 10178 if config['dataset']['name'] == 'CelebA' else len(dataset.class_to_idx)

    return dataset, unnormalizer, nb_classes


def training_steps(model, dataloader, unnormalize = None, classification = False, logger = None):
    device = torch.device('cuda' if next(model.parameters()).is_cuda else 'cpu')
    losses = []
    optimizer = configure_optimizer(model)
    loss_idx = 0
    for e in range(model.optim_params['epochs']):
        pbar = tqdm.tqdm(dataloader)
        
        for x, label in pbar:
            optimizer.zero_grad()
            out = model(x.to(device), label.to(device))
            loss_dict = model.loss_function(out, label.to(device)) if classification else model.loss_function(out, x.to(device),unnormalize)

            loss = loss_dict['loss']
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            if logger is not None:
                logger.add_scalar('loss',loss.item(), loss_idx)
                loss_idx += 1
            pbar.set_description(f'VAE epoch: %.3f Loss: %.3f' % (e,loss))

    if 'commitment_loss' in loss_dict.keys():
        # This means we have a VQ-VAE and we still need to train the model to generate into the latent space
        losses_generator = train_latent_generator(model, dataloader)
        
    return losses

def ELBO_gaussian(mu, logvar):
    return -torch.mean(0.5 * torch.sum(1-mu**2 - logvar.exp() +logvar, dim = 1), dim = 0)

@torch.no_grad
def evaluation_step(model, test_dataloader,kl_weight,unnormalize, classification = False, logger = None):
    device = torch.device('cuda' if next(model.parameters()).is_cuda else 'cpu')
    model.eval()
    loss = 0
    if classification:
        acc = 0
        
    for x, label in test_dataloader:
        out = model(x.to(device), label.to(device))
        loss_dict = model.loss_function(out, label.to(device)) if classification else model.loss_function(out, x.to(device), unnormalize)
        loss += loss_dict['loss'].item()
        if classification:
            acc += loss_dict['accuracy']

    if logger is not None:
        batch = next(iter(test_dataloader))
        with torch.no_grad():
            out = model(batch[0].to(device), batch[1].to(device))
        grid_in = torchvision.utils.make_grid(unnormalize(batch[0]), nrow = test_dataloader.batch_size // 2)
        grid_out = torchvision.utils.make_grid(unnormalize(out[0]).cpu(), nrow = test_dataloader.batch_size // 2)
        img_grid = torch.concat([grid_in, grid_out], dim = 1)
        logger.add_image('Test comparison', img_grid)

        logger.add_scalar('Loss eval', loss/len(test_dataloader.dataset))
        if classification:
            logger.add_scalar('Accuracy eval', acc/len(test_dataloader.dataset))

        if model.latent_dim == 2:
            batch, label = next(iter(test_dataloader))
            mu, logvar = model.encoder(batch.to(device))
            z = model.reparameterize(mu, logvar).cpu()
            plt.figure(1)
            plt.scatter(z[:,0], z[:,1], c= list(label.numpy()), cmap = 'viridis')
            plt.colorbar()
            logger.add_figure('Latent representation', plt.gcf())


    else:
        if classification:
            print(f'Accuracy: %.3f, Loss: %.3f' % (acc/len(test_dataloader.dataset), loss/len(test_dataloader.dataset)))
        else:
            print(f'Loss: %.3f' % ( loss/len(test_dataloader.dataset)))

def configure_optimizer(model):
    try:
        beta_1 = model.optim_params['beta_1']
    except:
        beta_1 = 0.9
    try:
        beta_2 = model.optim_params['beta_2']
    except:
        beta_2 = 0.999
    
    try:
        weight_decay = model.optim_params['weight_decay']
    except:
        weight_decay = 0

    optimizer = torch.optim.AdamW(model.parameters(), lr =model.optim_params['lr'], weight_decay=model.optim_params['weight_decay'], betas=(beta_1,beta_2))

    return optimizer

def train_latent_generator(model, dataloader):
    device = torch.device('cuda' if next(model.parameters()).is_cuda else 'cpu')
    to_save = None
    model.eval()
    with torch.no_grad():
        for im, _ in dataloader:
            im = im.to(device)
            _, _, _, quantized_idx = model(im)
            to_save = quantized_idx if to_save is None else torch.cat([to_save, quantized_idx], dim = 0)

        pickle.dump(to_save,open('./additional_datasets/mnist_quantized_encodings.pkl','wb'))
    
    encodings_mnist = MnistQuantizedEncoding(model.size_codebook, model.out_encoder_size)
    dataloader_encodings = DataLoader(encodings_mnist, batch_size = 64, shuffle = True)


    losses = []
    model.latent_generator.train()
    epochs = model.latent_generator.config['optimization']['epochs']
    optim = torch.optim.AdamW(model.latent_generator.parameters(),model.latent_generator.config['optimization']['lr'])
    for e in range(epochs):
        pbar = tqdm.tqdm(dataloader_encodings)
        for x in pbar:
            optim.zero_grad()
            out = model.latent_generator(x.float().to(device))
            loss_dict = model.latent_generator.loss_function(out, x.to(device))
            loss = loss_dict['loss']
            loss.backward()
            optim.step()
            losses.append(loss.item())

            pbar.set_description(f'Generator epoch: %.1f, loss: %.3f' % (e, loss.item()))
    return losses