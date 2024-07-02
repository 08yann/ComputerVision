import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import math
import copy
import abc
from utils import ELBO_gaussian
import tqdm
####################################################
### Compute output size of convolutions

## Conv2d:
# out = (in - kernel +2 * padding)// stride + 1

## ConvTranspose2d:
# out = (in - 1) * stride + kernel -2* padding + out_padding
####################################################

activation_dict = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),
            'id': nn.Identity()
        }

class ResBlock(nn.Module):
    def __init__(self, n_channel, kernel_size,stride = 1):
        super().__init__()
        
        if stride ==2:
            self.conv1 = nn.Conv2d(n_channel, n_channel * 2, kernel_size, stride = 2, padding = 1)
            self.n_channel = n_channel * 2
            self.downsample = True
            self.conv_downsample = nn.Conv2d(n_channel, self.n_channel,1, stride = 2)
        else:
            self.conv1 = nn.Conv2d(n_channel, n_channel, kernel_size, padding = 1)
            self.n_channel = n_channel
            self.downsample = False
        
        self.bnorm1 = nn.BatchNorm2d(self.n_channel, eps = 1e-8)
        self.conv2 = nn.Conv2d(self.n_channel, self.n_channel, kernel_size, padding = 1)
        self.bnorm2 = nn.BatchNorm2d(self.n_channel, eps = 1e-8)
        self.bnorm_skip = nn.BatchNorm2d(self.n_channel, eps = 1e-8)

    def forward(self,x, label = None):
        out = F.relu(self.bnorm1(self.conv1(x)))
        out = self.bnorm2(self.conv2(out))
        
        if self.downsample:
            x = self.bnorm_skip(self.conv_downsample(x))
        out = F.relu(out + x)
        return out

class ResNet(nn.Module):
    def __init__(self,config, img_shape, nb_classes):
        super().__init__()
        self.img_channels, self.img_size = img_shape[0], img_shape[1]
        self.nb_channels = config['model']['nb_channels']
        self.nb_blocks = config['model']['nb_blocks']
        self.size_block = config['model']['size_block']
        self.conv1 = nn.Conv2d(self.img_channels,self.nb_channels, 7, stride = 2, padding = 3)
        self.bnorm1 = nn.BatchNorm2d(self.nb_channels, eps = 1e-8)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride = 2, padding = 1) if config['model']['maxpool'] else nn.Identity()

        self.list_blocks = nn.ModuleList()
        for i in range(self.nb_blocks):
            for j in range(self.size_block):
                if j == 0:
                    self.list_blocks.append(ResBlock(2**i*self.nb_channels, kernel_size=3, stride = 2))
                else:
                    self.list_blocks.append(ResBlock(2**(i+1)*self.nb_channels, kernel_size=3, stride = 1))

        out_size = self.img_size
        halvings = self.nb_blocks + 1
        if config['model']['maxpool']: halvings += 1
        for _ in range(halvings):
            out_size = (out_size - 1)// 2 + 1
        self.out_size = out_size
        self.ln = nn.Linear(self.out_size**2*2**(self.nb_blocks)*self.nb_channels,nb_classes, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, label = None):
        B = x.shape[0]
        x = F.relu(self.bnorm1(self.conv1(x)))
        x = self.maxpool(x)

        for block in self.list_blocks:
            x = block(x)

        x = x.view(B,-1)
        x = self.ln(x)
        return x

    def loss_function(self, x_pred, x_true):
        loss = F.cross_entropy(x_pred, x_true)
        acc = (torch.argmax(x_pred, dim = -1) == x_true).float().sum()
        return {'loss': loss, 'accuracy': acc}


######################################
#  GAN
######################################
class Discriminator(nn.Module):
    def __init__(self, config, img_shape, nb_classes, conditionnal = False):
        super().__init__()
        nb_convs = len(config['model']['nb_channels'])
        assert nb_convs == len(config['model']['kernels'])
        assert nb_convs == len(config['model']['strides'])

        self.nb_classes = nb_classes
        self.conditionnal = conditionnal
        self.img_shape = img_shape
        nb_channels = copy.deepcopy(config['model']['nb_channels'])
        nb_channels.insert(0,img_shape[0])
        if self.conditionnal:
            nb_channels[0] += self.nb_classes
        self.out_size = img_shape[1]


        self.convs = nn.ModuleList()
        for i in range(nb_convs):
            pad = config['model']['kernels'][i]//2
            if config['model']['kernels'][i] % 2 == 0:
                pad -= 1
            self.out_size = (self.out_size - config['model']['kernels'][i]+ 2 *pad)// config['model']['strides'][i] + 1

            if i == nb_convs-2:
                self.start_gen_dim = self.out_size

            self.convs.append(nn.Sequential(
                nn.Conv2d(nb_channels[i], nb_channels[i+1], config['model']['kernels'][i], stride = config['model']['strides'][i], padding = pad),
                nn.BatchNorm2d(nb_channels[i+1]),
                nn.LeakyReLU(negative_slope = 0.2)
            ))
                            
        self.dropout = nn.Dropout(p = config['model']['dropout'])
        
        flatten_dim = self.out_size**2 * nb_channels[-1]

        self.head = nn.Linear(flatten_dim, 1, bias = False)
    
    def forward(self, x, label = None):
        B = x.shape[0]
        if self.conditionnal:
            label_one_hot = torch.zeros(B, self.nb_classes, self.img_shape[1], self.img_shape[2]).to(x.device)
            idx = torch.arange(B)
            label_one_hot[idx, label[idx].long()] = 1
            x = torch.cat([x,label_one_hot], dim = 1)

        for conv in self.convs:
            x = conv(x)
        x = self.dropout(x)
        x = x.reshape(B,-1)
        
        logits = F.sigmoid(self.head(x))
        return logits


class Generator(nn.Module):
    def __init__(self,config,img_shape,start_gen_dim,nb_classes, conditionnal = False):
        super().__init__()
        nb_convs = len(config['model']['nb_channels'])-1

        self.conditionnal = conditionnal
        self.nb_classes = nb_classes
        self.img_shape = img_shape
        self.start_gen_dim = start_gen_dim
        nb_channels = copy.deepcopy(config['model']['nb_channels'])[:-1]
        nb_channels = nb_channels[::-1]
        nb_channels.insert(0, nb_channels[0])

        self.latent_dim = config['model']['latent_dim']
        if self.conditionnal:
            self.latent_dim += self.nb_classes
        out_size = [img_shape[1]]
        output_pad = []
        for i in range(nb_convs):
            pad = config['model']['kernels'][i]//2
            if pad%2 == 0:
                pad-=1
            out_size.append((out_size[-1] - config['model']['kernels'][i]+ 2 *pad)// config['model']['strides'][i] + 1)
        sizes = out_size[::-1]

        self.output_pad = output_pad[::-1]
        self.init_convT = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, nb_channels[0], sizes[0], bias = False),
            nn.BatchNorm2d(nb_channels[0]),
            nn.LeakyReLU()
        )

        self.convTs = nn.ModuleList()
        for i in range(nb_convs):
            pad = config['model']['kernels'][-(i+2)]//2
            if pad%2 == 0:
                pad-= 1
            output_pad =  sizes[i+1] - ((sizes[i] - 1) * config['model']['strides'][-(i+2)] + config['model']['kernels'][-(i+2)] -2* pad)
            self.convTs.append(nn.Sequential(
                nn.ConvTranspose2d(nb_channels[i], nb_channels[i+1], config['model']['kernels'][-(i+2)], stride = config['model']['strides'][-(i+2)], padding = pad , output_padding = output_pad),
                nn.BatchNorm2d(nb_channels[i+1]),
                nn.LeakyReLU()
            ))
        self.out_conv = nn.Sequential(nn.ConvTranspose2d(nb_channels[-1], img_shape[0], kernel_size = 1, stride = 1), nn.Tanh())

        if self.conditionnal:
            self.latent_dim -= self.nb_classes

    def forward(self,x, label = None):
        B = x.shape[0]
        if self.conditionnal:
            label_one_hot = torch.zeros(B, self.nb_classes).to(label.device)
            label_one_hot.scatter_(1, label.unsqueeze(1).long(), 1)
            x = torch.cat([x, label_one_hot], dim = -1)

        x = x.view(B, self.latent_dim, 1, 1)
        x = self.init_convT(x)
        for convT in self.convTs:
            x = convT(x)

        x = self.out_conv(x)
        return x


def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

class GAN(nn.Module):
    def __init__(self, config, img_shape, nb_classes):
        super().__init__()
        self.optim_params = config['optimization']
        self.latent_dim = config['model']['latent_dim']
        self.img_shape = img_shape
        self.nb_classes = nb_classes
        try:
            self.conditionnal = config['model']['conditionnal']
        except:
            self.conditionnal = False

        self.discriminator = Discriminator(config, self.img_shape,self.nb_classes, self.conditionnal)
        self.generator = Generator(config,  self.img_shape, self.discriminator.start_gen_dim,self.nb_classes, self.conditionnal)
        
        self.discriminator.apply(weights_init)
        self.generator.apply(weights_init)

    def forward(self, x_real, label = None):
        B = x_real.shape[0]
        device = self.generator.ln.bias.device

        noise = torch.randn((B, self.latent_dim)).to(device)

        x_fake = self.generator(noise, label)
        x_cat = torch.cat([x_real, x_fake.detach()], dim = 0)

        out_discriminator = self.discriminator(x_cat,torch.cat([label, label], dim = 0))
        return out_discriminator,  x_fake

        loss_kl = ELBO_gaussian(x_out[1], x_out[2])
        loss = loss_mse + self.optim_params['kl_weight']*loss_kl
        return {'loss':loss, 'recon_loss':loss_mse, 'kl_loss':loss_kl}

    def loss_function(self, x_out, y):
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(x_out.squeeze(),y)
        return loss
    
    @torch.no_grad()
    def sample(self, nb_images, label = None):
        device = torch.device('cuda')
        if self.conditionnal:
            if label is None:
                label = torch.randint(0, self.nb_classes,(nb_images,)).to(device)
        z = torch.randn((nb_images, self.latent_dim)).to(device)
        return self.generator(z, label)

#########################################################
class BaseVAE(nn.Module):
    def __init__(self, device = 'cuda'):
        super().__init__()
    def encoder(self, x):
        raise NotImplementedError
    
    def decoder(self, z):
        raise NotImplementedError
    def reparameterization(self, mu, logvar):
        z = torch.randn_like(mu)
        std = torch.exp(0.5*logvar)
        return z*std + mu

    @abc.abstractmethod
    def forward(self, x, label = None):
        pass

    @torch.no_grad()
    def sample(self,nb_images, z = None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        z = torch.randn(10)
        raise NotImplementedError

    

# Variational Auto Encoder: https://arxiv.org/abs/1312.6114

class VAE_1D(nn.Module):
    def __init__(self,config, img_shape, nb_classes):
        super().__init__()
        self.img_shape = img_shape
        start_dim = img_shape[0]*img_shape[1]*img_shape[2]
        hidden_dim = copy.deepcopy(config['model']['encoder_dims'])
        hidden_dim.insert(0, start_dim)
        latent_dim = config['model']['latent_dim']
        self.encoder_layers = nn.ModuleList([nn.Sequential(
            nn.Linear(hidden_dim[i], hidden_dim[i+1]),
            nn.LeakyReLU()) for i in  range(len(hidden_dim)-1)])
        
        self.enc_mu = nn.Linear(hidden_dim[-1], latent_dim)
        self.enc_sigma = nn.Linear(hidden_dim[-1], latent_dim)

        self.decoder_init = nn.Linear(latent_dim, hidden_dim[-1])
        self.decoder_layer = nn.ModuleList([nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(hidden_dim[i], hidden_dim[i-1])) for i in range(len(hidden_dim)-1, 0, -1)])

                   
    def encoder(self, x):
        B = x.shape[0]
        x = x.view(B, -1)
        for layer in self.encoder_layers:
            x = layer(x)
        
        mu = self.enc_mu(x)
        logvar = self.enc_sigma(x)
        return mu, logvar
        
    def decoder(self, z):
        out = self.decoder_init(z)
        for layer in self.decoder_layer:
            out = layer(out)
        out = F.tanh(out).view(-1, self.img_shape[0],self.img_shape[1],self.img_shape[2])

        return out

    def reparameterize(self, mu, sig):
        eps = torch.randn_like(sig)
        std = torch.sqrt(torch.exp(sig))
        return mu + std*eps

    def forward(self, x, label = None):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)

        return x_hat, mu, logvar

class VAE_Block(nn.Module):
    def __init__(self,in_channel, out_channel, kernel, stride, last_activ = nn.LeakyReLU()):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel,stride = stride, padding = kernel//2, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=kernel, padding = kernel//2, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.downsample = None
        if stride != 1 or in_channel != out_channel:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride = stride, bias = False),
                nn.BatchNorm2d(out_channel))
        
        self.last_activation = last_activ
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x, label = None):
        if self.downsample is not None:
            resid = self.downsample(x)
        else:
            resid = x
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        x += resid
        out = self.last_activation(x)
        return out

class VAE_complex(nn.Module):
    def __init__(self, config, img_shape, nb_classes = None):
        super().__init__()

        self.conditionnal = config['model']['conditionnal']

        self.optim_params = config['optimization']
        self.img_shape = img_shape
        self.nb_classes = nb_classes
        nb_convs = len(config['model']['encoder_kernels'])
        assert len(config['model']['encoder_strides']) == nb_convs
        assert len(config['model']['encoder_channels']) == nb_convs

        nb_convT = len(config['model']['decoder_channels'])
        assert nb_convT <= nb_convs

        self.init_conv = nn.Sequential(
            nn.Conv2d(self.img_shape[0], config['model']['encoder_channels'][0], kernel_size = config['model']['encoder_kernels'][0], stride = config['model']['encoder_strides'][0], padding=config['model']['encoder_kernels'][0]//2, bias = False),
            nn.BatchNorm2d(config['model']['encoder_channels'][0]),
            nn.LeakyReLU())

        if config['model']['maxpool_stride']==0:
            self.maxpool = nn.Identity()
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride = config['model']['maxpool_stride'], padding = 1)
        
        self.encoder_blocks = nn.ModuleList()
        for i in range(1, nb_convs):
            self.encoder_blocks.append(VAE_Block(config['model']['encoder_channels'][i-1], config['model']['encoder_channels'][i], config['model']['encoder_kernels'][i], config['model']['encoder_strides'][i]))

            for j in range(1, config['model']['encoder_size_blocks']):
                self.encoder_blocks.append(VAE_Block(config['model']['encoder_channels'][i], config['model']['encoder_channels'][i], config['model']['encoder_kernels'][i], stride = 1))
        sizes_history = []
        img_size_out = img_shape[1]
        sizes_history.append(img_size_out)
        for i in range(nb_convs):
            img_size_out = (img_size_out - config['model']['encoder_kernels'][i] +2 * (config['model']['encoder_kernels'][i]//2))// config['model']['encoder_strides'][i] + 1
            sizes_history.append(img_size_out)
            if (i == 0) & (config['model']['maxpool_stride']>1):
                img_size_out = (img_size_out - 3 + 2)//config['model']['maxpool_stride'] + 1
                sizes_history.append(img_size_out)

        self.img_sizes_hist = sizes_history

        # reduce image to size with height and width: 1 x 1
        if config['model']['avgpool']:
            self.encoder_avgpool = nn.AvgPool2d(kernel_size = img_size_out, stride = 1)
            mlp_size = config['model']['encoder_channels'][-1]
        else:
            self.encoder_avgpool = nn.Identity()
            mlp_size = config['model']['encoder_channels'][-1]*self.img_sizes_hist[-1]**2
        
        self.mlp = nn.Linear(mlp_size, mlp_size)

        self.mlp_mu = nn.Sequential(
            nn.Linear(mlp_size, mlp_size),
            nn.LeakyReLU(),
            nn.Linear(mlp_size, config['model']['latent_dim']))

        self.mlp_logvar = nn.Sequential(
            nn.Linear(mlp_size, mlp_size),
            nn.LeakyReLU(),
            nn.Linear(mlp_size, config['model']['latent_dim']))

        # Decoder
        
        nb_convT = len(config['model']['decoder_channels'])
        self.decoder_output_pad = []

        in_size = img_shape[1]
        # Compute size of the necessary last linear layer before ConvTranspose in decoder to ensure output has same size as data
        for _ in range(nb_convT):
            if in_size % 2 == 0:
                in_size = in_size// 2
                self.decoder_output_pad.append(1)
            else:
                in_size = (in_size +1)//2
                self.decoder_output_pad.append(0)
        
        self.decoder_output_pad = self.decoder_output_pad[::-1]
    

        self.start_img_size = in_size
        self.latent_dim = config['model']['latent_dim']

        self.mlp_decoder = nn.Sequential(
            nn.Linear(self.latent_dim,self.latent_dim),
            nn.LeakyReLU(),
            nn.Linear(self.latent_dim, self.start_img_size**2*config['model']['decoder_channels'][0]),
            nn.LeakyReLU()
        )

        self.convT_layers = nn.ModuleList()
        for i in range(nb_convT):
            if i == nb_convT -1:
                self.convT_layers.append(nn.Sequential(
                    nn.ConvTranspose2d(config['model']['decoder_channels'][i], self.img_shape[0], 3, stride = 2, padding = 1, output_padding=self.decoder_output_pad[i])))
            else:
                self.convT_layers.append(nn.Sequential(
                    nn.ConvTranspose2d(config['model']['decoder_channels'][i], config['model']['decoder_channels'][i+1], 3, stride = 2, padding = 1, output_padding=self.decoder_output_pad[i]),
                    nn.LeakyReLU()
                ))

        self.convT_layers = nn.Sequential(*self.convT_layers)
        if config['model']['decoder_nb_blockconvs'] == 0:
            self.decoder_blockconvs = nn.ModuleList([nn.Tanh()])    
        else:
            self.decoder_blockconvs = nn.ModuleList([VAE_Block(self.img_shape[0],self.img_shape[0] ,kernel=3, stride = 1)  for _ in range(config['model']['decoder_nb_blockconvs']-1)])
            self.decoder_blockconvs.append(VAE_Block(self.img_shape[0],self.img_shape[0], kernel=3, stride = 1, last_activ=nn.Tanh()))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def reparameterize(self, mu, logvar):
        eps1 = torch.randn_like(logvar)
        eps2 = torch.randn_like(logvar)
        std = torch.exp(0.5*logvar)
        return mu + std* eps1

    def encoder(self, x):
        B = x.shape[0]
        x = self.init_conv(x)
        x = self.maxpool(x)

        for block in self.encoder_blocks:
            x = block(x)
        x = self.encoder_avgpool(x)
        x_encoded = x.view(B,-1)

        mu = self.mlp_mu(x_encoded)
        logvar = self.mlp_logvar(x_encoded)
        return mu, logvar

    def decoder(self,x, label = None):
        B = x.shape[0]
        x = self.mlp_decoder(x)

        x = x.view(B,-1,self.start_img_size,self.start_img_size )
        x = self.convT_layers(x)
        for block in self.decoder_blockconvs:
            x = block(x)
        return x

    def forward(self, x, label = None):
        B = x.shape[0]
        mu, logvar = self.encoder(x)

        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        
        return x_hat, mu, logvar
    
    def loss_function(self, x_out, x_true, unnormalize = None):
        try:
            reduc =  self.optim_params['reduction']
        except:
            reduc = 'mean' 
        if self.optim_params['criterion'] == 'bce':
            loss_mse = self.img_shape[1] * self.img_shape[0] *F.binary_cross_entropy(unnormalize(x_out[0]), unnormalize(x_true), reduction = 'none').mean(dim = (1,2,3))
        else:
            loss_mse = F.mse_loss(x_out[0], x_true, reduction=reduc)
        loss_kl = ELBO_gaussian(x_out[1], x_out[2])
        loss = (loss_mse + self.optim_params['kl_weight']*loss_kl).mean()
        return {'loss':loss, 'recon_loss':loss_mse, 'kl_loss':loss_kl}

    @torch.no_grad()
    def sample(self, nb_images, z = None):            
        if z is None:
            device = self.init_conv[0].weight.device
            z = torch.randn((nb_images, self.latent_dim)).to(device)

        x_pred = self.decoder(z)
        return x_pred

    def eval_manifold(self, unnormalize, nb_points = 900):
        assert nb_points == int(math.sqrt(nb_points)**2)

        list_grids = []
        if self.conditionnal:
            range_class = self.nb_classes
        else:
            range_class = 1
        
        for label_val in range(range_class):

            grid_side = int(math.sqrt(nb_points))
            xs = torch.linspace(-10,10,grid_side)
            ys = torch.linspace(-10,10,grid_side)

            xs, ys = torch.meshgrid([xs,ys])
            xs = xs.reshape(-1,1)
            ys = ys.reshape(-1,1)
            device = self.init_conv[0].weight.device
            zs = torch.cat([xs,ys], dim = -1).to(device)

            labels = torch.ones(nb_points,).to(device).long()* label_val
            generated_img = self.decoder(zs, labels)
            generated_img = ((generated_img.cpu() + 1)/2)
            grid = torchvision.utils.make_grid(generated_img, nrow = grid_side)
            list_grids.append(grid)

        if self.conditionnal:
            return list_grids

        return grid


class MultiStage_VAE(nn.Module):
    def __init__(self, config, img_shape, nb_classes):
        super().__init__()
        self.img_shape = img_shape
        self.optim_params = config['optimization']
        self.config = config

        self.vae = VAE_complex(self.config, img_shape, nb_classes)
        self.latent_dim = self.vae.latent_dim
        self.conditionnal = self.vae.conditionnal
        size_second_stage = self.config['model']['size_second_stage']
        self.second_stage = nn.ModuleList([VAE_Block(img_shape[0], img_shape[0],kernel=3, stride=1) for _ in range(size_second_stage-1)])
        self.second_stage.append(VAE_Block(img_shape[0], img_shape[0],kernel=3, stride=1, last_activ = nn.Tanh()))
        self.second_stage = nn.Sequential(*self.second_stage)
    
    def encoder(self, x):
        return self.vae.encoder(x)

    def forward(self,x, label = None):
        x_first, mu, logvar = self.vae(x)
        x_coarse = x_first
        x_coarse = x_coarse.detach()
        out = self.second_stage(x_coarse)

        return out, mu, logvar, x_first
    
    def reparameterize(self, mu, logvar):
        return self.vae.reparameterize(mu, logvar)
    def decoder(self, x):
        x = self.vae.decoder(x)
        return self.second_stage(x)

    @torch.no_grad()
    def sample(self, nb_images, z = None):            
        if z is None:
            device = self.second_stage[0].conv1.weight.device
            z = torch.randn((nb_images, self.vae.latent_dim)).to(device)

        x_first = self.vae.decoder(z)
        x_pred = self.second_stage(x_first)
        return x_pred
    
    def loss_function(self, x_out, x_true, unnormalize = None):
        try:
            reduc =  self.optim_params['reduction']
        except:
            reduc = 'mean' 
        if self.optim_params['criterion'] == 'bce':
            loss_mse = self.img_shape[1] * self.img_shape[0] *F.binary_cross_entropy(unnormalize(x_out[3]), unnormalize(x_true), reduction = 'none').mean(dim = (1,2,3))
        else:
            loss_mse = F.mse_loss(x_out[3], x_true, reduction=reduc)
        loss_kl = ELBO_gaussian(x_out[1], x_out[2])

        loss_l1 = F.l1_loss(x_out[0], x_true)
        loss = (loss_mse + self.optim_params['kl_weight']*loss_kl).mean() + loss_l1
        return {'loss':loss, 'recon_loss':loss_mse, 'kl_loss':loss_kl}

    @torch.no_grad()
    def eval_manifold(self, unnormalize , nb_points = 900):
        assert nb_points == int(math.sqrt(nb_points)**2)
        list_grids = []
        grid_side = int(math.sqrt(nb_points))
        xs = torch.linspace(-10,10,grid_side)
        ys = torch.linspace(-10,10,grid_side)

        xs, ys = torch.meshgrid([xs,ys])
        xs = xs.reshape(-1,1)
        ys = ys.reshape(-1,1)
        device = self.second_stage[0].conv1.weight.device
        zs = torch.cat([xs,ys], dim = -1).to(device)

        generated_img = self.decoder(zs)
        generated_img = unnormalize(generated_img).cpu()
        grid = torchvision.utils.make_grid(generated_img, nrow = grid_side)
        list_grids.append(grid)

        return grid


# Simple model, interesting for MNIST dataset especially the manifold function for 2-dimensional latent space. 
class VAE_3D(nn.Module):
    def __init__(self, config, img_shape, nb_classes = None):
        super().__init__()

        assert config['model']['name'] == 'VAE_3D'
        self.optim_params = config['optimization']
        self.img_channels = img_shape[0]
        self.img_size = img_shape[1]
        self.img_shape = img_shape
        self.config = copy.deepcopy(config['model'])

        assert len(self.config['encoder_kernels']) == len(self.config['encoder_strides'])
        assert len(self.config['encoder_kernels']) == len(self.config['encoder_channels'])

        self.nb_classes = nb_classes
        nb_convs = len(self.config['encoder_kernels'])

        self.conditionnal = self.config['conditionnal']

        if 'encoder_pad' in self.config:
            self.pad = self.config['encoder_pad']
        else:
            self.pad = [s//2 if s%2 == 1 else 0 for s in self.config['encoder_kernels']]

        self.nb_channels = copy.deepcopy(self.config['encoder_channels'])
        self.nb_channels.insert(0, self.img_channels)

        enc_mlp_dims = self.config['encoder_mlp_dims']
        self.latent_dim = self.config['encoder_mlp_dims'][-1]

        curr_size = self.img_size
        sizes_hist = []
        sizes_hist.append(curr_size)
        for i in range(len(self.config['encoder_kernels'])):
            curr_size = (curr_size- self.config['encoder_kernels'][i] + 2 * self.pad[i])//self.config['encoder_strides'][i] + 1
            sizes_hist.append(curr_size)

        reverse_sizes_hist = sizes_hist[::-1]
        enc_flat_dim = curr_size**2*self.nb_channels[-1]
        enc_mlp_dims.insert(0, enc_flat_dim)
        dec_mlp_dims = enc_mlp_dims[::-1]

        self.sizes_hist = sizes_hist

        self.output_pad = [0 if (reverse_sizes_hist[i+1] - self.config['encoder_kernels'][-(i+1)]+ 2* self.pad[-(i+1)]) % self.config['encoder_strides'][-(i+1)] == 0 else 1 for i in range(nb_convs) ]
        self.dec_kernels = self.config['encoder_kernels'][::-1]


        if self.conditionnal:
            dec_mlp_dims[0] += self.nb_classes
        
        for i in range(len(self.dec_kernels)):
            if reverse_sizes_hist[i+1] != (reverse_sizes_hist[i]-1)*self.config['encoder_strides'][-i-1]-2*self.pad[-i-1] + self.dec_kernels[i] + self.output_pad[i]:
                self.dec_kernels[i] += 1
                self.output_pad[i] = 0 if (reverse_sizes_hist[i+1] - self.dec_kernels[i]+ 2* self.pad[-(i+1)]) % self.config['encoder_strides'][-(i+1)] == 0 else 1
        
        self.enc_convs = nn.ModuleList([nn.Sequential(nn.Conv2d(self.nb_channels[i], self.nb_channels[i+1], kernel_size=self.config['encoder_kernels'][i], stride=self.config['encoder_strides'][i], padding = self.pad[i]),
                                        nn.BatchNorm2d(self.nb_channels[i+1]),
                                        activation_dict[self.config['encoder_activation']]) for i in range(nb_convs)])

        self.mlp_mu = nn.ModuleList([nn.Sequential(nn.Linear(enc_mlp_dims[i], enc_mlp_dims[i+1]), activation_dict[self.config['encoder_mlp_activation']] if i<len(enc_mlp_dims)-2 else nn.Identity()) for i in range(len(enc_mlp_dims)-1)])
        self.mlp_logvar = nn.ModuleList([nn.Sequential(nn.Linear(enc_mlp_dims[i], enc_mlp_dims[i+1]), activation_dict[self.config['encoder_mlp_activation']] if i<len(enc_mlp_dims)-2 else nn.Identity()) for i in range(len(enc_mlp_dims)-1)])

        self.dec_convs = nn.ModuleList([nn.Sequential(
                                            nn.ConvTranspose2d(self.nb_channels[-i-1], self.nb_channels[-i-2], kernel_size=self.dec_kernels[i],stride=self.config['encoder_strides'][-i-1], padding=self.pad[-i-1], output_padding=self.output_pad[i]),
                                            nn.BatchNorm2d(self.nb_channels[-i-2]),
                                        nn.Tanh() if i == nb_convs-1 else activation_dict[self.config['decoder_activation']]) for i in range(nb_convs)])

        #Ensure last activation leads to values in [0,1] 
        self.dec_mlp = nn.ModuleList([nn.Sequential(nn.Linear(dec_mlp_dims[i], dec_mlp_dims[i+1]), activation_dict[self.config['decoder_mlp_activation']] ) for i in range(len(dec_mlp_dims)-1)])
        
    def reparameterize(self, mu, logvar):
        eps1 = torch.randn_like(logvar)
        std = torch.exp(0.5*logvar)
        return mu + std* eps1

    def decoder(self, z, label = None):
        if self.conditionnal:
            label_one_hot = torch.zeros(z.shape[0], self.nb_classes).to(label.device)
            label_one_hot.scatter_(1, label.unsqueeze(1).long(), 1)
            z = torch.cat([z, label_one_hot], dim = -1)

        for fc in self.dec_mlp:
            z = fc(z)
        
        size_deflat = int(math.sqrt(z.shape[-1]/self.nb_channels[-1]))
        x_hat = z.reshape(z.shape[0], self.nb_channels[-1], size_deflat, size_deflat)
        for conv in self.dec_convs:
            x_hat= conv(x_hat)
        
        return x_hat
    def encoder(self, x, label = None):
        b = x.shape[0]

        for conv in self.enc_convs:
            x = conv(x)
        
        out = x.reshape(b,-1)

        mu = out
        logvar = out
        for fc in self.mlp_mu:
            mu = fc(mu)

        for fc in self.mlp_logvar:
            logvar = fc(logvar)
        return mu, logvar

    def forward(self, x, label = None):
        b = x.shape[0]

        mu, logvar = self.encoder(x,label)
                
        z = self.reparameterize(mu, logvar)

        x_hat = self.decoder(z, label)

        return x_hat, mu, logvar

    def loss_function(self, x_out, x_true, unnormalize = None):
        try:
            reduc =  self.optim_params['reduction']
        except:
            reduc = 'mean' 
        if self.optim_params['criterion'] == 'bce':
            loss_mse = self.img_shape[1] * self.img_shape[0] *F.binary_cross_entropy(unnormalize(x_out[0]), unnormalize(x_true), reduction = 'none').mean(dim = (1,2,3))
        else:
            loss_mse = F.mse_loss(x_out[0], x_true, reduction=reduc)
        loss_kl = ELBO_gaussian(x_out[1], x_out[2])
        loss = (loss_mse + self.optim_params['kl_weight']*loss_kl).mean()
        return {'loss':loss, 'recon_loss':loss_mse, 'kl_loss':loss_kl}

    @torch.no_grad()
    def sample(self, nb_images, z = None, label = None):            
        if z is None:
            device = self.enc_convs[0][0].bias.device
            z = torch.randn((nb_images, self.latent_dim)).to(device)
        if self.conditionnal:
            if label is None:
                label = torch.randint(0, self.nb_classes,(nb_images,)).to(device)

        x_pred = self.decoder(z, label)
        return x_pred


    @torch.no_grad()
    def eval_manifold(self, unnormalize, nb_points = 900):
        assert nb_points == int(math.sqrt(nb_points)**2)

        list_grids = []
        if self.conditionnal:
            range_class = self.nb_classes
        else:
            range_class = 1
        
        for label_val in range(range_class):

            grid_side = int(math.sqrt(nb_points))
            xs = torch.linspace(-10,10,grid_side)
            ys = torch.linspace(-10,10,grid_side)

            xs, ys = torch.meshgrid([xs,ys])
            xs = xs.reshape(-1,1)
            ys = ys.reshape(-1,1)
            device = self.enc_convs[0][0].bias.device
            zs = torch.cat([xs,ys], dim = -1).to(device)

            labels = torch.ones(nb_points,).to(device).long()* label_val
            generated_img = self.decoder(zs, labels)
            generated_img = unnormalize(generated_img).cpu()
            grid = torchvision.utils.make_grid(generated_img, nrow = grid_side)
            list_grids.append(grid)

        if self.conditionnal:
            return list_grids

        return grid


class Quantizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config['model']['size_codebook'],config['model']['latent_dim'])

    
    def forward(self, x, label = None):
        B, C, H, W = x.shape
        x = x.permute(0,2,3,1)
        x = x.reshape(x.size(0), -1, x.size(-1))

        dist = torch.cdist(x, self.embedding.weight[None, :].repeat(x.shape[0],1,1))
        idx_emb = torch.argmin(dist, dim = -1)
        quant = torch.index_select(self.embedding.weight, 0, idx_emb.view(-1))

        x = x.reshape(-1, x.size(-1))

        loss_embed = torch.mean((x.detach() - quant)**2)
        loss_commitment = torch.mean((x-quant.detach())**2)

        # Trick to ensure only gradient of decoder output is passed in backward pass.
        quant_out = x + (quant -x).detach()

        quant_out = quant_out.reshape(B, H, W, C)

        quant_out = quant_out.permute(0,3,1,2)
        return quant_out, loss_embed, loss_commitment, idx_emb
    
    def embed_from_quantized_idx(self, quantized_idx):
        return quantized_idx

class  VQ_VAE(nn.Module):
    def __init__(self, config, img_shape, nb_classes):
        super().__init__()
        self.nb_convs = len(config['model']['encoder_kernels'])
        self.optim_params = config['optimization']
        self.latent_dim = config['model']['latent_dim']
        self.size_codebook = config['model']['size_codebook']
        self.generator_name = config['latent_generator']['model']['name']
        self.conditionnal = config['model']['conditionnal']
        self.img_shape = img_shape
        self.nb_classes = nb_classes
        assert len(config['model']['encoder_strides']) == self.nb_convs
        assert len(config['model']['encoder_channels']) == self.nb_convs

        nb_channels = copy.deepcopy(config['model']['encoder_channels'])
        nb_channels.insert(0, self.img_shape[0])
    
        self.pad = [config['model']['encoder_kernels'][i]//2 if config['model']['encoder_kernels'][i]%2 != 0 else 0 for i in range(self.nb_convs)]
        sizes_hist = []
        sizes_hist.append(self.img_shape[1])
        for i in range(self.nb_convs):
            sizes_hist.append((sizes_hist[-1]-config['model']['encoder_kernels'][i]+2*self.pad[i])//config['model']['encoder_strides'][i] + 1)

        self.out_encoder_size = sizes_hist[-1]

        reverse_sizes_hist = sizes_hist[::-1]

        self.output_pad = [0 if (reverse_sizes_hist[i+1] - config['model']['encoder_kernels'][-(i+1)]+ 2* self.pad[-(i+1)]) % config['model']['encoder_strides'][-(i+1)] == 0 else 1 for i in range(self.nb_convs)]

        
        self.enc_convs = nn.ModuleList([nn.Sequential(nn.Conv2d(nb_channels[i], nb_channels[i+1], kernel_size=config['model']['encoder_kernels'][i], stride=config['model']['encoder_strides'][i], padding=self.pad[i]),
                                                    nn.BatchNorm2d(nb_channels[i+1]),
                                                    activation_dict[config['model']['encoder_activation']]) for i in range(self.nb_convs-1)])
        
        self.enc_convs.append(nn.Conv2d(nb_channels[-2], nb_channels[-1], kernel_size=config['model']['encoder_kernels'][-1],stride = config['model']['encoder_strides'][-1], padding=self.pad[-1]))
            
        self.dec_kernels = copy.deepcopy(config['model']['encoder_kernels'][::-1])
        for i in range(len(self.dec_kernels)):
            if reverse_sizes_hist[i+1] != (reverse_sizes_hist[i]-1)*config['model']['encoder_strides'][-i-1]-2*self.pad[-i-1] + self.dec_kernels[i] + self.output_pad[i]:
                self.dec_kernels[i] += 1
                self.output_pad[i] = 0 if (reverse_sizes_hist[i+1] - self.dec_kernels[i]+ 2* self.pad[-(i+1)]) % config['model']['encoder_strides'][-(i+1)] == 0 else 1

        self.dec_convs = nn.ModuleList([nn.Sequential(nn.ConvTranspose2d(nb_channels[-i-1], nb_channels[-i-2], kernel_size=self.dec_kernels[i], stride=config['model']['encoder_strides'][-i-1], padding=self.pad[-i-1], output_padding=self.output_pad[i]),
                                                    nn.BatchNorm2d(nb_channels[-i-2]),
                                                    activation_dict[config['model']['decoder_activation']]) for i in range(self.nb_convs - 1)])

        self.dec_convs.append(nn.Sequential(nn.ConvTranspose2d(nb_channels[1], nb_channels[0], kernel_size=self.dec_kernels[-1], stride=config['model']['encoder_strides'][0], padding=self.pad[0], output_padding=self.output_pad[-1]),nn.Tanh()))


        self.pre_conv_quant = nn.Conv2d(nb_channels[-1], self.latent_dim, kernel_size=1)
        self.post_conv_quant = nn.Conv2d(self.latent_dim, nb_channels[-1], kernel_size = 1)
        self.quantizer = Quantizer(config)


        self.latent_generator = self.configure_latent_model(config['latent_generator'])

    def configure_latent_model(self, config):
        dict_latent_models = {'LSTM': LSTM_VQVAE, 'PixelCNN': PixelCNN_VQVAE}
        assert config['model']['name'] in dict_latent_models
        return dict_latent_models[config['model']['name']](config, (1, self.out_encoder_size, self.out_encoder_size), self.size_codebook)

    def encoder(self, x):
        for conv in self.enc_convs:
            x = conv(x)
        return x


    def decoder(self, z):
        for convT in self.dec_convs:
            z = convT(z)
        return z

    def forward(self, x, label = None):
        enc_out = self.encoder(x)
        enc_out = self.pre_conv_quant(enc_out)

        quant_out, loss_embed, loss_commitment, quantized_idx = self.quantizer(enc_out)
        quant_out = self.post_conv_quant(quant_out)
        out = self.decoder(quant_out)
        return out, loss_embed, loss_commitment, quantized_idx

    def loss_function(self, x_out, x_true, unnormalize = None):
        try:
            reduc =  self.optim_params['reduction']
        except:
            reduc = 'mean' 
        if self.optim_params['criterion'] == 'bce':
            loss_mse = F.binary_cross_entropy(unnormalize(x_out[0]), unnormalize(x_true), reduction = reduc)
        else:
            loss_mse = F.mse_loss(x_out[0], x_true, reduction=reduc)

        loss_commitment = x_out[2]
        loss_embed = x_out[1]
        loss = loss_mse + self.optim_params['commitment_weight']*loss_commitment + loss_embed
        return {'loss':loss, 'recon_loss':loss_mse, 'commitment_loss':loss_commitment, 'embed_loss': loss_embed}

    @torch.no_grad()
    def sample(self, nb_images, z = None, label = None):            
        if z is None:
            device = self.enc_convs[0][0].bias.device
            z = self.latent_generator.generate(nb_images)

        embedded_batch = self.quantizer.embedding(z.long().squeeze(1)).permute(0,3,1,2)
        x_pred = self.decoder(self.post_conv_quant(embedded_batch))
        
        return x_pred

class LSTM_VQVAE(nn.Module):
    def __init__(self, config,img_shape, nb_classes):
        super().__init__()
        self.optim_params = config['optimization']
        self.latent_dim = config['model']['latent_dim_generator']
        self.context_size_dataset = config['model']['context_size_dataset']
        self.size_codebook = nb_classes
        self.img_shape = img_shape
        hidden_dim = config['model']['hidden_dim']
        nb_layers = config['model']['nb_layers']
        self.lstm = nn.LSTM(input_size=self.latent_dim, hidden_size=hidden_dim, num_layers=nb_layers, batch_first = True)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//4), nn.ReLU(), nn.Linear(hidden_dim//4, self.size_codebook + 2))
        self.word_embedding = nn.Embedding(self.size_codebook + 2, self.latent_dim)

    def forward(self,x , label = None):
        x = self.word_embedding(x)
        out, _= self.lstm(x)
        #out = out[:,-1,:]
        return self.mlp(out)

    def loss_function(self, x_out, x_true):
        loss = torch.mean(F.cross_entropy(x_out.permute(0,2,1), x_true.long()))
        return {'loss':loss}
    
    @torch.no_grad()
    def generate(self, nb_samples):
        quantized_idx = []
        encodings_len = self.img_shape[1]*self.img_shape[2]

        device = self.mlp[0].bias.device

        for _ in range(nb_samples):
            seq = torch.ones(1).to(device)*self.size_codebook
            for i in range(encodings_len):
                padded_seq = seq 
                if len(padded_seq) < self.context_size_dataset:
                    padded_seq = nn.functional.pad(padded_seq, pad = (0,self.context_size_dataset-len(padded_seq)),value = self.size_codebook)
                out = self(padded_seq[-self.context_size_dataset:].unsqueeze(0).long().to(device))
                if i >= self.context_size_dataset:
                    probs = F.softmax(out[0][-1], dim = -1)
                else:
                    probs = F.softmax(out[0][i], dim = -1)
                idx = torch.multinomial(probs, num_samples = 1)
                if idx >= self.size_codebook:
                    idx = torch.multinomial(probs[:-2], num_samples = 1)
                seq = torch.cat([seq, idx])
            quantized_idx.append(seq[1:].unsqueeze(0))
        quantized_batch = torch.cat(quantized_idx, dim = 0)
        quantized_batch = quantized_batch.reshape(quantized_batch.shape[0], self.img_shape[1],self.img_shape[2]).long().unsqueeze(1)
        return quantized_batch




class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d,self).__init__(*args, **kwargs)

        assert mask_type in ['A', 'B'], 'Invalid type of mask'

        self.mask_type = mask_type

        self.register_buffer('mask', self.weight.data.clone())

        H, W = self.weight.shape[2], self.weight.shape[3]

        self.mask.fill_(1)

        if self.mask_type == 'A':
            self.mask[:,:,H//2, W//2:] = 0
            self.mask[:,:,H//2+1:, :] = 0
        elif self.mask_type == 'B':
            self.mask[:,:,H//2, W//2 + 1:] = 0
            self.mask[:,:,H//2+ 1:, :] = 0
    
    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


class PixelCNN(nn.Module):
    def __init__(self, config, img_shape, nb_classes):
        super().__init__()
        self.nb_out = config['model']['nb_out']
        self.kernel = config['model']['kernel']
        self.channel_dim = config['model']['channel_dim']
        self.optim_params = config['optimization']

        self.conv_init = MaskedConv2d('A',1, 2*self.channel_dim, kernel_size = 7, padding = 3)

        self.blocks = nn.ModuleList()
        mask = 'B'
        for _ in range(config['model']['nb_layers']):
            self.blocks.append(nn.Sequential(
                nn.ReLU(),
                MaskedConv2d(mask, 2*self.channel_dim, self.channel_dim, kernel_size = 1),
                nn.BatchNorm2d(self.channel_dim),
                nn.ReLU(),
                MaskedConv2d(mask, self.channel_dim, self.channel_dim, kernel_size = self.kernel, padding = self.kernel//2),
                nn.BatchNorm2d(self.channel_dim),
                nn.ReLU(),
                MaskedConv2d(mask, self.channel_dim, 2*self.channel_dim, kernel_size = 1),
                nn.BatchNorm2d(2*self.channel_dim)
            ))
            
        self.conv_out = nn.Sequential(
            nn.ReLU(),
            MaskedConv2d(mask, 2*self.channel_dim, self.channel_dim, kernel_size = 1),
            nn.ReLU(),
            MaskedConv2d(mask, self.channel_dim, self.channel_dim, kernel_size = 1),
            MaskedConv2d(mask, self.channel_dim, self.nb_out, kernel_size = 1)
        )

        if self.nb_out == 2:
            self.conv_out.append(nn.Sigmoid())
        else:
            self.conv_out.append(nn.Softmax(dim = 1))

    def forward(self, x, label = None):
        x = self.conv_init(x)

        for conv in self.blocks:
            x = conv(x) + x
        
        out = self.conv_out(x)
    
        return out

    def loss_function(self, x_out, x_true, unnormalize):
        try:
            reduc =  self.optim_params['reduction']
        except:
            reduc = 'mean' 
        if self.optim_params['criterion'] == 'bce':
            loss_mse = F.binary_cross_entropy(x_out, unnormalize(x_true).long().float())
        else:
            loss_mse = F.mse_loss(x_out[0], x_true, reduction=reduc)
        loss = loss_mse
        return {'loss':loss, 'recon_loss':loss_mse}


    @torch.no_grad()
    def generate(self, nb_samples, img_size):
        img = torch.zeros(nb_samples, 1, img_size, img_size).to(self.conv_out[1].bias.device)
        for i in range(img_size):
            for j in range(img_size):
                logits = self(2*(img-.5))
                if self.nb_out == 2:
                    img[:,:,i,j] = torch.bernoulli(logits[:, :, i, j], out=logits[:, :, i, j])
                else:
                    img[:,:,i,j] = torch.multinomial(logits[:, :, i, j], num_samples=1)/(self.nb_out-1)

        return img


class PixelCNN_VQVAE(nn.Module):
    def __init__(self, config, img_shape,nb_classes):
        super().__init__()
        self.optim_params = config['optimization']
        self.size_codebook = nb_classes
        self.img_shape = img_shape
        nb_layers = config['model']['nb_layers']
        channel_dim = config['model']['channel_dim']
        kernels = [3 if i>0 else 7 for i in range(nb_layers)]
        masks = ['B' if i>0 else 'A' for i in range(nb_layers)]

        channels = [channel_dim] * nb_layers
        channels.insert(0, img_shape[0])
        self.convs = nn.ModuleList()

        for i in range(nb_layers):
            self.convs.append(nn.Sequential(MaskedConv2d(masks[i], channels[i], channels[i+1], kernels[i], padding = kernels[i]//2),
                                            nn.BatchNorm2d(channels[i+1]),
                                            nn.ReLU()))

        self.conv_out = nn.Conv2d(channels[-1], self.size_codebook, kernel_size = 1)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            if i > 0:
                x = x + conv(x)
            else:
                x = conv(x)
        out = self.conv_out(x)
        return out

    def loss_function(self, x_out, x_true, unnormalize = None):
        try:
            criterion = self.optim_params['criterion']
        except:
            criterion = 'mse'
        if criterion == 'cross_entropy':
            loss = F.cross_entropy(x_out, x_true.long().squeeze(1))
        else:
            loss = F.mse_loss(x_out, x_true)
        return {'loss':loss}
        
        
    @torch.no_grad()
    def generate(self, nb_samples):
        img = torch.zeros(nb_samples,self.img_shape[0],  self.img_shape[1], self.img_shape[2]).to(self.conv_out.bias.device)
        for i in range(self.img_shape[1]):
            for j in range(self.img_shape[2]):
                logits = self(img)
                probs = F.softmax(logits, dim = 1)

                img[:,:,i,j] = torch.multinomial(probs[:,:,i,j] , num_samples = 1)
        return img