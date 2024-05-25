import torch
import torch.nn.functional as F
import torch.nn as nn

# Supervised contrastive loss: https://arxiv.org/pdf/2004.11362
# Dot-product of high-dimensional representation of the classifier output, then compute softmax and loss function is minus log-likelihood.

# Simple MNIST experiment
class PixelNormLayer(nn.Module):
    def __init__(self, eps = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        x = x*torch.rsqrt(torch.mean(x, dim = 1, keepdim=True) + self.eps)
        return x

class AddNoise(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels))
    
    def forward(self,x):
        noise = torch.randn(x.shape[0], 1, x.shape[2],x.shape[3], device=x.device)
        x = x + self.weight.view(1,-1,1,1) * noise
        x = F.leaky_relu(x, negative_slope=0.2)
        return x

class AdaIN(nn.Module):
    def __init__(self, latent_dim, channels):
        super().__init__()
        self.style_lin = nn.Linear(latent_dim, channels*2)
        self.norm = nn.InstanceNorm2d(channels)

    def forward(self, x, w):
        x = self.norm(x)
        styles = self.style_lin(w)
        styles = styles.view(x.shape[0], 2, x.shape[1],1,1)
        # In below operation, we add 1.0 to ensure that scaling doesn't start at zero.
        x = x * (1.0 + styles[:,0]) + styles[:,1]
        return x

class InitSynthesisBlock(nn.Module):
    def __init__(self, latent_dim, channels = 64, start_size = 4):
        super().__init__()
        self.init_constant = nn.Parameter(torch.ones(1,channels, start_size,start_size))
        self.bias_init = nn.Parameter(torch.ones(channels))

        self.add_noise = AddNoise(channels)

        self.ada_in = AdaIN(latent_dim, channels)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        self.add_noise_2 = AddNoise(channels)
        self.ada_in_2 = AdaIN(latent_dim, channels)

    def forward(self, w):
        B = w.shape[0] # batch size
        x = self.init_constant.expand(B, -1,-1,-1)
        x =x + self.bias_init.view(1,-1,1,1)
        x = self.add_noise(x)
        x = self.ada_in(x,w)
        x = self.conv(x)
        x = self.add_noise_2(x)
        x = self.ada_in_2(x,w)
        return x

class SynthesisBlock(nn.Module):
    def __init__(self, latent_dim, channels):
        super().__init__()
        self.up_conv = nn.Conv2d(channels, channels//2, kernel_size=3, padding=1)
        
        self.add_noise = AddNoise(channels//2)

        self.ada_in = AdaIN(latent_dim, channels//2)
        self.conv = nn.Conv2d(channels//2, channels//2, kernel_size=3, padding=1)

        self.add_noise_2 = AddNoise(channels//2)
        self.ada_in_2 = AdaIN(latent_dim, channels//2)
            
    def forward(self,x,w):
        x = x.view(x.shape[0], x.shape[1], x.shape[2], 1,x.shape[3],1).expand(-1,-1,-1,2,-1,2)
        x = x.contiguous().view(x.shape[0], x.shape[1], x.shape[2]*2,x.shape[4]*2)
        x = self.up_conv(x)
        x = self.add_noise(x)
        x = self.ada_in(x,w)
        x = self.conv(x)
        x = self.add_noise_2(x)
        x = self.ada_in_2(x,w)
        return x

## StyleGAN: https://arxiv.org/abs/1812.04948 and Github: https://github.com/huangzh13/StyleGAN.pytorch

class MappingNetwork(nn.Module):
    def __init__(self, latent_dim, nb_layers):
        super().__init__()
        self.latent_dim = latent_dim

        self.fc = nn.ModuleList()
        self.fc.append(nn.Sequential(nn.Linear(self.latent_dim, latent_dim), nn.LeakyReLU(negative_slope=0.2)))
        for _ in range(1,nb_layers-1):
            self.fc.append(nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.LeakyReLU(negative_slope=0.2)
            ))

        self.fc.append(nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.LeakyReLU(negative_slope=0.2)))
        
    
    def forward(self, x):
        for mlp in self.fc:
            x = mlp(x)
        return x

class SynthesisNetwork(nn.Module):
    def __init__(self,img_channel, latent_dim, init_channel, start_size, nb_blocks = 2):
        super().__init__()
        self.init = InitSynthesisBlock(latent_dim, init_channel, start_size)

        self.blocks = nn.ModuleList()
        for i in range(nb_blocks):
            self.blocks.append(SynthesisBlock(latent_dim, init_channel//(2**i)))
        
        self.out_conv = nn.Conv2d(init_channel//(2**(nb_blocks)), img_channel, kernel_size = 1)
    
    def forward(self,w):
        x = self.init(w)
        for block in self.blocks:
            x = block(x,w)
        x = F.tanh(self.out_conv(x))
        return x

class StyleGenerator(nn.Module):
    def __init__(self, latent_dim, nb_layers_mapping, img_channel = 1, init_channel = 64, start_size = 7, nb_blocks = 2):
        super().__init__()
        self.mapping = MappingNetwork(latent_dim, nb_layers_mapping)
        self.synthesis = SynthesisNetwork(img_channel, latent_dim, init_channel, start_size, nb_blocks)

    def forward(self, z):
        w = self.mapping(z)
        x = self.synthesis(w)
        return x


class BlurLayer(nn.Module):
    def __init__(self, kernel = None, stride = 1):
        super().__init__()
        if kernel is None:
            kernel = [[1,2,1],[2,4,2], [1,2,1]]
            kernel = torch.tensor(kernel).unsqueeze(0).unsqueeze(0)
            kernel = kernel/kernel.sum()
        self.register_buffer('kernel', kernel)
        self.stride = stride

    def forward(self,x):
        kernel = self.kernel.expand(x.shape[1],-1,-1,-1)
        x = F.conv2d(x, kernel, stride = self.stride, padding=(self.kernel.shape[-1]-1)//2,groups = x.shape[1])
        return x
    

class Discriminator(nn.Module):
    def __init__(self, img_channel, start_channel, nb_blocks = 2):
        super().__init__()
        self.init_conv = nn.Conv2d(img_channel, start_channel, kernel_size=1)

        self.blocks = nn.ModuleList()
        for i in range(nb_blocks):
            channel = 2**i*start_channel
            self.blocks.append(nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=3, padding=1),
                nn.LeakyReLU(negative_slope=0.2),
                BlurLayer(),
                nn.Conv2d(channel, 2*channel, kernel_size=3, padding = 1),
                BlurLayer(kernel = torch.tensor([[[[0.2500, 0.2500],[0.2500, 0.2500]]]]), stride = 2),
                #nn.AvgPool2d(kernel_size=2),
                nn.LeakyReLU(negative_slope=0.2)
            ))

        self.final_block = nn.Sequential(
            nn.Conv2d(2*channel, 2*channel, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.final_linear = nn.Sequential(
            nn.Linear(2*channel * 7 * 7, 2*channel),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(2*channel, 1)
        )

    def forward(self, x):
        x = self.init_conv(x)
        for block in self.blocks:
            x = block(x)
        
        x = self.final_block(x)
        x = x.view(x.shape[0],-1)
        x = self.final_linear(x)
        return x

class StyleGAN(nn.Module):
    def __init__(self, config, img_shape, nb_classes):
        super().__init__()
        self.img_channel = img_shape[0]
        self.latent_dim = config['model']['latent_dim']
        self.discriminator = Discriminator(self.img_channel, config['model']['start_channel'], config['model']['nb_blocks_discr'])
        self.generator = StyleGenerator(config['model']['latent_dim'], config['model']['nb_layers_mapping'], self.img_channel , config['model']['init_channel'], config['model']['start_size'], nb_blocks = 2)
        self.optim_params = config['optimization']

        self._get_gen_optimizer()
        self._get_discr_optimizer()

    
    def _get_gen_optimizer(self):
        self.optim_generator = torch.optim.Adam(self.generator.parameters(), lr = self.optim_params['lr_generator'], betas = (self.optim_params['gen_beta_1'],0.99))

    def _get_discr_optimizer(self):
        self.optim_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr = self.optim_params['lr_discriminator'], betas = (self.optim_params['discr_beta_1'],0.99), eps = 1e-8)


    def discriminator_loss(self, x_fake, x_real):
        device = x_fake.device
        criterion = nn.BCEWithLogitsLoss()

        out_fake = self.discriminator(x_fake)

        out_real = self.discriminator(x_real)

        loss_real = criterion(out_real.squeeze(),torch.ones(out_real.shape[0]).to(device))
        loss_fake = criterion(out_fake.squeeze(),torch.zeros(out_fake.shape[0]).to(device))

        loss = (loss_real + loss_fake)/2

        return loss

    def generator_loss(self, x):
        device = x.device
        out = self.discriminator(x)
        criterion = nn.BCEWithLogitsLoss()

        return criterion(out.squeeze(), torch.ones(out.shape[0]).to(device))

    
    def discriminator_step(self,noise, x_real):
        x_fake = self.generator(noise)
        loss = self.discriminator_loss(x_fake, x_real)

        self.optim_discriminator.zero_grad()
        loss.backward()
        self.optim_discriminator.step()

        return loss.item()
    
    def generator_step(self, noise):
        x_fake = self.generator(noise)
        loss = self.generator_loss(x_fake)
        self.optim_generator.zero_grad()
        loss.backward()
        self.optim_generator.step()

        return loss.item()












'''


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel = 3, stride = 2):
        self.conv0 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel, padding=kernel//2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=kernel//2, stride = stride)

        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride = stride, padding = kernel//2, bias=False)

    def forward(self, x):
        x_skip = self.skip(x)

        x = F.leaky_relu(self.conv0(x))
        x = F.leaky_relu(self.conv1(x))
        x += x_skip
        return x


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_dim = 100
        self.ln_start = nn.Linear(self.latent_dim, 49*128)

        self.convT = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride = 2, padding = 1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64,1,3,stride = 2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )

        self.out_conv = nn.Conv2d(1,1,3, padding=1)       


    def forward(self,x):
        x = self.ln_start(x)
        x = x.view(x.shape[0], 128,7,7)
        x = self.convT(x)
        x = F.tanh(self.out_conv(x))
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride = 2, padding=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64,128, 3, stride = 2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
        )


        self.ln = nn.Linear(49*128, 1, bias = False)
    
    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.shape[0], -1)
        x = self.ln(x)
        return x

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.init = nn.Conv2d(1, 32, kernel_size=1)

        self.blocks = nn.Sequential(
            Block(32, 64),
            Block(64,128)
        )

        self.conv_head = nn.Conv2d(128, 128, 3, padding=1)
        self.fc1 = nn.Linear(128*49, 128)
        self.fc2 = nn.Linear(128, 512)
        self.out = nn.Linear(512,10)
        
    def forward(self,x):
        x = F.leaky_relu(self.init(x))
        x = self.blocks(x)

        x = F.leaky_relu(self.conv_head(x))
        x = F.leaky_relu(self.fc1(x.flatten(1)))
        x = self.fc2(x)
        x = self.out(x)
        return x

class DuDGAN(nn.Module):
    def __init__(self, config, img_shape, nb_classes = 0):
        super().__init__()
        self.discriminator = Discriminator()

        self.generator = Generator()
        self.classifier = Classifier()

    def forward(self, x):
        return x

'''