import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import math
import copy

####################################################
### Compute output size of convolutions

## Conv2d:
# out = (in - kernel +2 * padding)// stride + 1

## ConvTranspose2d:
# out = (in - 1) * stride + kernel -2* padding
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
        
        self.bnorm1 = nn.BatchNorm2d(self.n_channel, eps = 1e-8, momentum = 0)
        self.conv2 = nn.Conv2d(self.n_channel, self.n_channel, kernel_size, padding = 1)
        self.bnorm2 = nn.BatchNorm2d(self.n_channel, eps = 1e-8, momentum = 0)
        self.bnorm_skip = nn.BatchNorm2d(self.n_channel, eps = 1e-8, momentum = 0)

    def forward(self,x):
        out = F.relu(self.bnorm1(self.conv1(x)))
        out = self.bnorm2(self.conv2(out))
        
        if self.downsample:
            x = self.bnorm_skip(self.conv_downsample(x))
        out = F.relu(out + x)

        return out


class ResNet(nn.Module):
    def __init__(self,n_classes, nb_channels = 64, nb_repeat = 2, nb_blocks = 4):
        super().__init__()

        self.conv1 = nn.Conv2d(3,nb_channels, 7, stride = 2, padding = 3)
        self.bnorm1 = nn.BatchNorm2d(nb_channels, eps = 1e-8, momentum = 0)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride = 2, padding = 1)

        list_blocks = []
        for _ in range(nb_repeat):
            list_blocks.append(ResBlock(nb_channels, 3))

        for i in range(1, nb_blocks):
            for j in range(nb_repeat):
                if j == 0:
                    list_blocks.append(ResBlock(2**(i-1)*nb_channels, 3, 2))
                else:
                    list_blocks.append(ResBlock(2**i*nb_channels,3))
            
        self.blocks = nn.Sequential(*list_blocks)

        self.ln = nn.Linear(7*7*2**i*nb_channels,n_classes, bias=False)


    def forward(self, x):
        B = x.shape[0]
        x = F.relu(self.bnorm1(self.conv1(x)))
        x = self.maxpool(x)

        for block in self.blocks:
            x = block(x)

        x = x.view(B,-1)
        x = self.ln(x)
        return x

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)



######################################
#  GAN
######################################

class Discriminator(nn.Module):
    def __init__(self, img_channel,img_size = 224, nb_channels = 64, drop = 0.2):
        super().__init__()

        self.conv1 = nn.Conv2d(img_channel, nb_channels, kernel_size=7, stride = 2, padding = 3)
        self.conv2 = nn.Conv2d(nb_channels, nb_channels*2,kernel_size=3, stride = 2,padding = 1)
        self.bn1 = nn.BatchNorm2d(2*nb_channels)
        self.conv3 = nn.Conv2d(nb_channels*2, nb_channels*4, kernel_size=3, stride = 2, padding=1)
        self.bn2 = nn.BatchNorm2d(4*nb_channels)
        self.conv4 = nn.Conv2d(4*nb_channels, 4*nb_channels, kernel_size=3, stride = 2, padding = 1)
        self.bn3 = nn.BatchNorm2d(nb_channels*4)
        self.conv5 = nn.Conv2d(4*nb_channels, 8*nb_channels, kernel_size= 3, stride = 2, padding=1)

        self.dropout = nn.Dropout(p = drop)
        
        flatten_dim = (img_size // (2**5))**2 * 2**3*nb_channels

        self.head = nn.Linear(flatten_dim, 1, bias = False)
    
    def forward(self, x):
        B = x.shape[0]
        x = F.leaky_relu(self.conv1(x), negative_slope = 0.2)
        x = F.leaky_relu(self.bn1(self.conv2(x)), negative_slope = 0.2)
        x = F.leaky_relu(self.bn2(self.conv3(x)), negative_slope = 0.2)
        x = F.leaky_relu(self.bn3(self.conv4(x)), negative_slope = 0.2)
        x = F.leaky_relu(self.conv5(x), negative_slope = 0.2)
        x = self.dropout(x)
        x = x.reshape(B, -1)
        logits = F.sigmoid(self.head(x))
        return logits


class Generator(nn.Module):
    def __init__(self,latent_dim,img_channel = 3,  nb_channels = 64, img_compressed = 7):
        super().__init__()
        self.img_compressed = img_compressed
        self.channels_start = 8*nb_channels
        self.hidden_dim = self.img_compressed**2*self.channels_start

        self.ln = nn.Linear(latent_dim, self.hidden_dim)
        self.conv1T = nn.ConvTranspose2d(8*nb_channels, 4*nb_channels,kernel_size=3, stride = 2, padding = 1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(4*nb_channels)
        self.conv2T = nn.ConvTranspose2d(4*nb_channels, 4*nb_channels,kernel_size=3, stride = 2, padding = 1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(4*nb_channels)
        self.conv3T = nn.ConvTranspose2d(4*nb_channels, 2*nb_channels,kernel_size=3, stride = 2, padding = 1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(2*nb_channels)
        self.conv4T = nn.ConvTranspose2d(2*nb_channels, nb_channels,kernel_size=3, stride = 2, padding = 1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(nb_channels)
        self.conv5T = nn.ConvTranspose2d(nb_channels, img_channel, kernel_size= 7, stride = 2, padding = 3, output_padding=1)


    def forward(self,x):
        B = x.shape[0]
        x = F.relu(self.ln(x))
        x = x.view(B, self.channels_start, self.img_compressed, self.img_compressed)

        x = F.relu(self.bn1(self.conv1T(x)))
        x = F.relu(self.bn2(self.conv2T(x)))
        x = F.relu(self.bn3(self.conv3T(x)))
        x = F.relu(self.bn4(self.conv4T(x)))
        x = F.tanh(self.conv5T(x))

        return x




# Variational Auto Encoder: https://arxiv.org/abs/1312.6114

class VAE(nn.Module):
    def __init__(self, img_size, hidden_dim,  latent_dim):
        super().__init__()
        self.enc_l1 = nn.Linear(img_size ** 2, hidden_dim)
        self.enc_l2 = nn.Linear(hidden_dim, hidden_dim)
        self.enc_mu = nn.Linear(hidden_dim, latent_dim)
        self.enc_sigma = nn.Linear(hidden_dim, latent_dim)

        self.dec_l1 = nn.Linear(latent_dim, hidden_dim)
        self.dec_l2 = nn.Linear(hidden_dim, img_size**2)

        
    def encoder(self, x):
        h = F.leaky_relu(self.enc_l1(x), negative_slope=0.2)
        h = F.leaky_relu(self.enc_l2(h), negative_slope=0.2)

        mu = self.enc_mu(h)
        logvar = self.enc_sigma(h)

        return mu, logvar
        
    def decoder(self, z):
        out = F.leaky_relu(self.dec_l1(z), negative_slope=0.2)
        out = F.tanh(self.dec_l2(out))
        return out

    def reparameterization(self, mu, sig):
        eps = torch.randn_like(sig)
        std = torch.sqrt(torch.exp(sig))
        return mu + std*eps

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterization(mean, logvar)

        x_hat = self.decoder(z)

        return x_hat, mean, logvar
    


class VAE_3D(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert len(config['encoder_kernels']) == len(config['encoder_strides'])
        assert len(config['encoder_kernels']) == len(config['encoder_channels'])

        self.nb_classes = config['nb_classes']
        nb_convs = len(config['encoder_kernels'])

        self.conditionnal = config['conditionnal']

        if 'encoder_pad' in config:
            self.pad = config['encoder_pad']
        else:
            self.pad = [s//2 if s%2 == 1 else 0 for s in config['encoder_kernels']]

        self.nb_channels = copy.deepcopy(config['encoder_channels'])
        self.nb_channels.insert(0, config['img_channels'])

        enc_mlp_dims = config['encoder_mlp_dims']
        self.latent_dim = config['encoder_mlp_dims'][-1]

        curr_size = config['img_size']
        sizes_hist = []
        sizes_hist.append(curr_size)
        for i in range(len(config['encoder_kernels'])):
            curr_size = (curr_size- config['encoder_kernels'][i] + 2 * self.pad[i])//config['encoder_strides'][i] + 1
            sizes_hist.append(curr_size)

        reverse_sizes_hist = sizes_hist[::-1]
        enc_flat_dim = curr_size**2*self.nb_channels[-1]
        enc_mlp_dims.insert(0, enc_flat_dim)
        dec_mlp_dims = enc_mlp_dims[::-1]

        self.output_pad = [0 if (reverse_sizes_hist[i+1] - config['encoder_kernels'][-(i+1)]+ 2* self.pad[-(i+1)]) % config['encoder_strides'][-(i+1)] == 0 else 1 for i in range(nb_convs) ]
        self.dec_kernels = config['encoder_kernels'][::-1]


        if self.conditionnal:
            self.nb_channels[0] += self.nb_classes
            dec_mlp_dims[0] += self.nb_classes
        
        for i in range(len(self.dec_kernels)):
            if reverse_sizes_hist[i+1] != (reverse_sizes_hist[i]-1)*config['encoder_strides'][-i-1]-2*self.pad[-i-1] + self.dec_kernels[i] + self.output_pad[i]:
                self.dec_kernels[i] += 1
                self.output_pad[i] = 0 if (reverse_sizes_hist[i+1] - self.dec_kernels[i]+ 2* self.pad[-(i+1)]) % config['encoder_strides'][-(i+1)] == 0 else 1
        
        self.enc_convs = nn.ModuleList([nn.Sequential(nn.Conv2d(self.nb_channels[i], self.nb_channels[i+1], kernel_size=config['encoder_kernels'][i], stride=config['encoder_strides'][i], padding = self.pad[i]),
                                        nn.BatchNorm2d(self.nb_channels[i+1]),
                                        activation_dict[config['encoder_activation']]) for i in range(nb_convs)])


        self.mlp_mu = nn.ModuleList([nn.Sequential(nn.Linear(enc_mlp_dims[i], enc_mlp_dims[i+1]), activation_dict[config['encoder_mlp_activation']]) for i in range(len(enc_mlp_dims)-1)])
        self.mlp_logvar = nn.ModuleList([nn.Sequential(nn.Linear(enc_mlp_dims[i], enc_mlp_dims[i+1]), activation_dict[config['encoder_mlp_activation']]) for i in range(len(enc_mlp_dims)-1)])
        
        if self.conditionnal:
            self.nb_channels[0] -= self.nb_classes
        self.dec_convs = nn.ModuleList([nn.Sequential(
                                            nn.ConvTranspose2d(self.nb_channels[-i-1], self.nb_channels[-i-2], kernel_size=self.dec_kernels[i],stride=config['encoder_strides'][-i-1], padding=self.pad[-i-1], output_padding=self.output_pad[i]),
                                        nn.BatchNorm2d(self.nb_channels[-i-2]),
                                        nn.Tanh() if i == nb_convs-1 else activation_dict[config['decoder_activation']]) for i in range(nb_convs)])

        #Ensure last activation leads to values in [0,1] 
        self.dec_mlp = nn.ModuleList([nn.Sequential(nn.Linear(dec_mlp_dims[i], dec_mlp_dims[i+1]), activation_dict[config['decoder_mlp_activation']] ) for i in range(len(dec_mlp_dims)-1)])
        
        
    def reparameterize(self, mu, logvar):
        eps1 = torch.randn_like(logvar)
        eps2 = torch.randn_like(logvar)
        std = torch.exp(0.5*logvar)
        return mu + std* eps1

    def decoder(self, z, label = None):
        if self.conditionnal:
            device = self.enc_convs[0][0].bias.device
            label_one_hot = torch.zeros(z.shape[0], self.nb_classes).to(device)
            batch_idx = torch.arange(0, z.shape[0])
            label_one_hot[batch_idx, label] = 1
            z = torch.cat([z, label_one_hot], dim = -1)

        for fc in self.dec_mlp:
            z = fc(z)
        
        size_deflat = int(math.sqrt(z.shape[-1]/self.nb_channels[-1]))
        x_hat = z.reshape(z.shape[0], self.nb_channels[-1], size_deflat, size_deflat)
        for conv in self.dec_convs:
            x_hat= conv(x_hat)
        
        return x_hat


    def forward(self, x, label = None):
        b = x.shape[0]

        if self.conditionnal:
            device = self.enc_convs[0][0].bias.device
            label_expanded = torch.zeros(b, self.nb_classes, *x.shape[2:]).to(device)
            batch_idx = torch.arange(0, b, device = device)
            label_expanded[batch_idx, label,:,:] = 1

            x = torch.cat([x,label_expanded], dim = 1)


        for conv in self.enc_convs:
            x = conv(x)
        
        out = x.reshape(b,-1)

        mu = out
        logvar = out
        for fc in self.mlp_mu:
            mu = fc(mu)

        for fc in self.mlp_logvar:
            logvar = fc(logvar)
        
        z = self.reparameterize(mu, logvar)

        x_hat = self.decoder(z)

        return x_hat, mu, logvar

    @torch.no_grad()
    def sample(self, nb_images, z = None, label = None):            
        if z is None:
            device = self.enc_convs[0][0].bias.device
            z = torch.randn((nb_images, self.latent_dim)).to(device)

        x_pred = self.decoder(z, label)
        return x_pred


    @torch.no_grad()
    def eval_manifold(self, nb_points = 900):
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
            generated_img = ((generated_img.cpu() + 1)/2)
            grid = torchvision.utils.make_grid(generated_img, nrow = grid_side)
            list_grids.append(grid)

        if self.conditionnal:
            return list_grids

        return grid


class Quantizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config['size_codebook'],config['latent_dim'])

    
    def forward(self, x):
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
    def __init__(self, config):
        super().__init__()
        self.nb_convs = len(config['encoder_kernels'])

        assert len(config['encoder_strides']) == self.nb_convs
        assert len(config['encoder_channels']) == self.nb_convs

        nb_channels = copy.deepcopy(config['encoder_channels'])
        nb_channels.insert(0, config['img_channels'])
    
        self.pad = [config['encoder_kernels'][i]//2 if config['encoder_kernels'][i]%2 != 0 else 0 for i in range(self.nb_convs)]
        sizes_hist = []
        sizes_hist.append(config['img_size'])
        for i in range(self.nb_convs):
            sizes_hist.append((sizes_hist[-1]-config['encoder_kernels'][i]+2*self.pad[i])//config['encoder_strides'][i] + 1)

        self.out_encoder_size = sizes_hist[-1]

        reverse_sizes_hist = sizes_hist[::-1]

        self.output_pad = [0 if (reverse_sizes_hist[i+1] - config['encoder_kernels'][-(i+1)]+ 2* self.pad[-(i+1)]) % config['encoder_strides'][-(i+1)] == 0 else 1 for i in range(self.nb_convs)]

        
        self.enc_convs = nn.ModuleList([nn.Sequential(nn.Conv2d(nb_channels[i], nb_channels[i+1], kernel_size=config['encoder_kernels'][i], stride=config['encoder_strides'][i], padding=self.pad[i]),
                                                    nn.BatchNorm2d(nb_channels[i+1]),
                                                    activation_dict[config['encoder_activation']]) for i in range(self.nb_convs-1)])
        
        self.enc_convs.append(nn.Conv2d(nb_channels[-2], nb_channels[-1], kernel_size=config['encoder_kernels'][-1],stride = config['encoder_strides'][-1], padding=self.pad[-1]))
            
        self.dec_kernels = copy.deepcopy(config['encoder_kernels'][::-1])
        for i in range(len(self.dec_kernels)):
            if reverse_sizes_hist[i+1] != (reverse_sizes_hist[i]-1)*config['encoder_strides'][-i-1]-2*self.pad[-i-1] + self.dec_kernels[i] + self.output_pad[i]:
                self.dec_kernels[i] += 1
                self.output_pad[i] = 0 if (reverse_sizes_hist[i+1] - self.dec_kernels[i]+ 2* self.pad[-(i+1)]) % config['encoder_strides'][-(i+1)] == 0 else 1

        self.dec_convs = nn.ModuleList([nn.Sequential(nn.ConvTranspose2d(nb_channels[-i-1], nb_channels[-i-2], kernel_size=self.dec_kernels[i], stride=config['encoder_strides'][-i-1], padding=self.pad[-i-1], output_padding=self.output_pad[i]),
                                                    nn.BatchNorm2d(nb_channels[-i-2]),
                                                    activation_dict[config['decoder_activation']]) for i in range(self.nb_convs - 1)])

        self.dec_convs.append(nn.Sequential(nn.ConvTranspose2d(nb_channels[1], nb_channels[0], kernel_size=self.dec_kernels[-1], stride=config['encoder_strides'][0], padding=self.pad[0], output_padding=self.output_pad[-1]),nn.Tanh()))


        self.pre_conv_quant = nn.Conv2d(nb_channels[-1], config['latent_dim'], kernel_size=1)
        self.post_conv_quant = nn.Conv2d(config['latent_dim'], nb_channels[-1], kernel_size = 1)
        self.quantizer = Quantizer(config)

    def encoder(self, x):
        for conv in self.enc_convs:
            x = conv(x)
        return x


    def decoder(self, z):
        for convT in self.dec_convs:
            z = convT(z)
        return z

    def forward(self, x):
        enc_out = self.encoder(x)
        enc_out = self.pre_conv_quant(enc_out)

        quant_out, loss_embed, loss_commitment, quantized_idx = self.quantizer(enc_out)
        quant_out = self.post_conv_quant(quant_out)
        out = self.decoder(quant_out)
        return out, loss_embed, loss_commitment, quantized_idx


class LSTM_VQVAE(nn.Module):
    def __init__(self, config, hidden_dim = 256, nb_layers = 3):
        super().__init__()
        self.latent_dim = config['latent_dim']
        self.size_codebook = config['size_codebook']
        self.lstm = nn.LSTM(input_size=self.latent_dim, hidden_size=hidden_dim, num_layers=nb_layers, batch_first = True)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//4), nn.ReLU(), nn.Linear(hidden_dim//4, self.size_codebook))
        self.word_embedding = nn.Embedding(self.size_codebook + 2, self.latent_dim)

    def forward(self,x ):
        x = self.word_embedding(x)
        out, _= self.lstm(x)
        out = out[:,-1,:]
        return self.mlp(out)
        

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


# First try for PixelCNN
'''
class PixelCNN(nn.Module):
    def __init__(self, in_channel, hidden_channel, nb_blocks, nb_classes = 1):
        super().__init__()
        self.nb_classes = nb_classes
        
        
        self.conv_init = MaskedConv2d('A', in_channel, hidden_channel, 7, padding = 3)
        self.convs = nn.ModuleList()
        mask = 'B'
        for i in range(nb_blocks):
            self.convs.append(nn.Sequential(MaskedConv2d(mask,hidden_channel,hidden_channel , kernel_size = 7, padding = 3),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(hidden_channel)))

        self.conv_out = nn.Sequential(MaskedConv2d(mask, hidden_channel, hidden_channel, kernel_size = 1),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(hidden_channel),
                                        MaskedConv2d(mask, hidden_channel, 1 , 1),
                                        nn.Sigmoid())
                                    

    def forward(self, x):
        x = self.conv_init(x)
        for conv in self.convs:
            x = conv(x)
        
        out = self.conv_out(x)

        return out

    @torch.no_grad()
    def generate(self, nb_samples, img_size):
        img = torch.zeros(nb_samples, 1, img_size, img_size).to(self.conv_out[0].bias.device)
        for i in range(img_size):
            for j in range(img_size):
                logits = self(2*(img-.5))
                img[:,:,i,j] = torch.bernoulli(logits[:, :, i, j], out=logits[:, :, i, j])

        return img'''

class PixelCNN(nn.Module):
    def __init__(self, nb_blocks = 15, h = 32, nb_classes = 2, kernel = 3):
        super().__init__()
        self.nb_classes = nb_classes

        self.conv_init = MaskedConv2d('A',1, 2*h, kernel_size = 7, padding = 3)

        self.blocks = nn.ModuleList()
        mask = 'B'
        for _ in range(nb_blocks):
            self.blocks.append(nn.Sequential(
                nn.ReLU(),
                MaskedConv2d(mask, 2*h, h, kernel_size = 1),
                nn.BatchNorm2d(h),
                nn.ReLU(),
                MaskedConv2d(mask, h, h, kernel_size = kernel, padding = kernel//2),
                nn.BatchNorm2d(h),
                nn.ReLU(),
                MaskedConv2d(mask, h, 2*h, kernel_size = 1),
                nn.BatchNorm2d(2*h)
            ))
            
        self.conv_out = nn.Sequential(
            nn.ReLU(),
            MaskedConv2d(mask, 2*h, h, kernel_size = 1),
            nn.ReLU(),
            MaskedConv2d(mask, h, h, kernel_size = 1),
            MaskedConv2d(mask, h, self.nb_classes, kernel_size = 1)
        )

        if self.nb_classes == 2:
            self.conv_out.append(nn.Sigmoid())
        else:
            self.conv_out.append(nn.Softmax(dim = 1))

    def forward(self, x):
        x = self.conv_init(x)

        for conv in self.blocks:
            x = conv(x) + x
        
        out = self.conv_out(x)
    
        return out

    @torch.no_grad()
    def generate(self, nb_samples, img_size):
        img = torch.zeros(nb_samples, 1, img_size, img_size).to(self.conv_out[1].bias.device)
        for i in range(img_size):
            for j in range(img_size):
                logits = self(2*(img-.5))
                if self.nb_classes == 2:
                    img[:,:,i,j] = torch.bernoulli(logits[:, :, i, j], out=logits[:, :, i, j])
                else:
                    img[:,:,i,j] = torch.multinomial(logits[:, :, i, j], num_samples=1)/(self.nb_classes-1)

        return img


class PixelCNN_VQVAE(nn.Module):
    def __init__(self, config,nb_layers = 12, channel_dim = 32):
        super().__init__()
        self.size_codebook = config['size_codebook']

        kernels = [3 if i>0 else 7 for i in range(nb_layers)]
        masks = ['B' if i>0 else 'A' for i in range(nb_layers)]

        channels = [channel_dim] * nb_layers
        channels.insert(0, 1)
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

    @torch.no_grad()
    def generate(self, nb_samples, img_size):
        img = torch.zeros(nb_samples,1,  img_size, img_size).to(self.conv_out.bias.device)
        for i in range(img_size):
            for j in range(img_size):
                logits = self(img)
                probs = F.softmax(logits, dim = 1)

                img[:,:,i,j] = torch.multinomial(probs[:,:,i,j] , num_samples = 1)
        return img