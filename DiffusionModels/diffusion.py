import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
import math
import tqdm

"https://github.com/bot66/MNISTDiffusion"
'''
paper Shuffle-Net v2, add encoder block separating channels in two, 
on one half we perform same operations as usual residual net but 
then we concat with the first half of the input channels 
following a procedure one from input, then one from residualnet and so on
'''
class ChannelShuffle(nn.Module):
    def __init__(self, nb_groups):
        super().__init__()
        self.nb_groups = nb_groups
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        x = x.view(B, self.nb_groups, C//self.nb_groups, H, W)
        x = x.transpose(1,2).contiguous().view(B, -1, H, W)
        return x


class ConvBnSilu(nn.Module):
    def __init__(self,in_channels, out_channels, kernel, stride = 1, padding = 0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride = stride, padding = padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.silu(self.bn(self.conv(x)))
        return out


class TimeMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.l1 = nn.Linear(in_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x,t):
        t = self.l2(F.silu(self.l1(t)))
        t = t.unsqueeze(-1).unsqueeze(-1)
        x = F.silu(x + t)
        return x

class ResidualBlottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel = 3):
        super().__init__()
        self.branch1 = nn.Sequential(nn.Conv2d(in_channels//2, in_channels//2, kernel, padding = 1, groups = in_channels//2),
                                        nn.BatchNorm2d(in_channels//2),
                                        ConvBnSilu(in_channels//2, out_channels//2, kernel = 1))
        self.branch2 = nn.Sequential(ConvBnSilu(in_channels//2, in_channels//2, kernel = 1),
                                        nn.Conv2d(in_channels//2, in_channels//2, kernel, padding = 1,groups = in_channels//2),
                                        nn.BatchNorm2d(in_channels//2),
                                        ConvBnSilu(in_channels//2, out_channels//2, kernel = 1))

        self.channel_shuffle = ChannelShuffle(2)
    
    def forward(self, x):
        x1, x2 = x.chunk(2, dim = 1)
        out = torch.cat([self.branch1(x1), self.branch2(x2)], dim = 1)
        out = self.channel_shuffle(out)
        return out


class ResidualDownsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel = 3):
        super().__init__()
        self.branch1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel, stride = 2, padding = 1, groups = in_channels),
                                        nn.BatchNorm2d(in_channels),
                                        ConvBnSilu(in_channels, out_channels//2, kernel = 1))
        self.branch2 = nn.Sequential(ConvBnSilu(in_channels, out_channels//2, kernel = 1),
                                        nn.Conv2d(out_channels//2, out_channels//2, kernel,stride = 2, padding = 1,groups = out_channels//2),
                                        nn.BatchNorm2d(out_channels//2),
                                        ConvBnSilu(out_channels//2, out_channels//2, kernel = 1))

        self.channel_shuffle = ChannelShuffle(2)
    
    def forward(self, x):
        out = torch.cat([self.branch1(x), self.branch2(x)], dim = 1)
        out = self.channel_shuffle(out)
        return out

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.residblock = nn.Sequential(*[ResidualBlottleneck(in_channels, in_channels) for i in range(3)], ResidualBlottleneck(in_channels, out_channels//2))
        self.conv_downsample = ResidualDownsample(out_channels//2, out_channels)
        self.time_mlp = TimeMLP(time_emb_dim, hidden_dim = out_channels, out_dim = out_channels//2)

    def forward(self, x, t = None):
        x_resid = self.residblock(x)
        if t is not None:
            x = self.time_mlp(x_resid, t)

        x = self.conv_downsample(x)

        return x, x_resid

class DecoderBlock(nn.Module):
    def __init__(self,in_channels, out_channels,time_emb_dim, kernel = 3):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners=False)
        self.residblock = nn.Sequential(*[ResidualBlottleneck(in_channels, in_channels) for i in range(3)],ResidualBlottleneck(in_channels,in_channels//2))

        self.conv = ResidualBlottleneck(in_channels//2, out_channels//2)

        self.time_mlp = TimeMLP(time_emb_dim, hidden_dim=in_channels, out_dim = in_channels//2)

    def forward(self,x, x_resid, t = None):
        x = self.upsample(x)
        x = torch.cat([x, x_resid], dim = 1)
        x = self.residblock(x)
        if t is not None:
            x = self.time_mlp(x, t)

        x = self.conv(x)
        
        return x


class UNet(nn.Module):
    def __init__(self, max_timesteps,time_emb_dim, img_dim = 28, img_channel = 1, base_channel = 64, channels_multiplier = [2,4], final_channels = 2):
        super().__init__()
        self.init_conv = ConvBnSilu(img_channel, base_channel, kernel = 3, stride = 1, padding = 1)

        self.time_embed = nn.Embedding(max_timesteps, time_emb_dim)


        assert isinstance(channels_multiplier,(list, tuple))
        self.channels_dim = [(channels_multiplier[i]*base_channel, channels_multiplier[i+1]*base_channel) for i in range(len(channels_multiplier)-1)]
        self.channels_dim.insert(0, (base_channel, base_channel*channels_multiplier[0]))

        self.encoder = nn.ModuleList([EncoderBlock(c[0],c[1],time_emb_dim) for c in self.channels_dim])
        self.decoder = nn.ModuleList([DecoderBlock(c[1],c[0],time_emb_dim) for c in self.channels_dim[::-1]])

        self.bottom = nn.Sequential(*[ResidualBlottleneck(self.channels_dim[-1][1], self.channels_dim[-1][1]) for i in range(2)], ResidualBlottleneck(self.channels_dim[-1][1], self.channels_dim[-1][1]//2))

        self.head = nn.Conv2d(base_channel//2,final_channels,1)

    def forward(self, x, t = None):
        x = self.init_conv(x)
        if t is not None:
            t_emb = self.time_embed(t)

        encoder_residuals = []
        for enc_block in self.encoder:
            x, x_resid = enc_block(x,t_emb)
            encoder_residuals.append(x_resid)
        
        x = self.bottom(x)
        for dec_block, resid in zip(self.decoder, encoder_residuals[::-1]):
            x = dec_block(x, resid, t_emb)
        
        x = self.head(x)
        return x

class MNISTDiffusion(nn.Module):
    def __init__(self, img_size, img_channel, time_emb_dim = 256, max_timesteps = 1000, base_channel = 32, channels_multiplier = [1,2,4,8]):
        super().__init__()
        self.img_channel = img_channel
        self.model = UNet(max_timesteps, time_emb_dim, img_dim=img_size, img_channel=img_channel, base_channel= base_channel, channels_multiplier=channels_multiplier, final_channels = img_channel)

        self.max_timesteps = max_timesteps
        self.img_size = img_size

        betas = self._cosine_schedule(max_timesteps)

        alphas = 1.-betas
        alphas_cumprod = torch.cumprod(alphas, dim = -1)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.-alphas_cumprod))
        

    def forward(self, x, noise):
        t = torch.randint(0, self.max_timesteps, (x.shape[0],)).to(x.device)
        xt = self._forward_diffusion(x,t,noise)
        pred_noise = self.model(xt, t)
        return pred_noise
        

    def _cosine_schedule(self, max_timesteps,  eps = 0.008):
        steps = torch.linspace(0, max_timesteps +1,steps = max_timesteps+1,dtype = torch.float32)
        ft = torch.cos(((steps/max_timesteps+eps)/(1.0 + eps))*math.pi*0.5)**2
        betas = torch.clip(1.0-ft[1:]/ft[:max_timesteps],0.0,0.999)

        return betas

    def _forward_diffusion(self, x0, t, noise):
        assert noise.shape == x0.shape
        xt = self.alphas_cumprod.gather(-1,t).reshape(x0.shape[0],1,1,1)*x0 + self.sqrt_one_minus_alphas_cumprod.gather(-1,t).reshape(x0.shape[0],1,1,1)*noise
        return xt

    @torch.no_grad()
    def sampling(self, nb_samples, device, clipped_reverse_diffusion = True):
        xt = torch.randn((nb_samples, self.img_channel, self.img_size, self.img_size)).to(device)
        for i in tqdm.tqdm(range(self.max_timesteps-1,-1,-1), desc = 'Sampling'):
            noise = torch.randn_like(xt).to(device)
            t = torch.tensor([i for _ in range(nb_samples)], dtype = torch.int64).to(device)

            if clipped_reverse_diffusion:
                xt = self._reverse_diffusion_clipped(xt, t,noise)
            else:
                xt = self._reverse_diffusion(xt,t,noise)
        
        xt = (xt+1.)/2.
        return xt

    @torch.no_grad()
    def _reverse_diffusion(self, xt, t, noise):
        noise_pred = self.model(xt,t)

        alpha_t = self.alphas.gather(-1,t).reshape(xt.shape[0],1,1,1)
        alpha_t_cumprod = self.alphas_cumprod.gather(-1,t).reshape(xt.shape[0],1,1,1)


        mean = 1./torch.sqrt(alpha_t)* (xt-(1.-alpha_t)/torch.sqrt(1.-alpha_t_cumprod)*noise_pred)

        if t.min() > 0:
            alpha_t_minus_1_cumprod = self.alphas_cumprod.gather(-1,t).reshape(xt.shape[0],1,1,1)
            std = torch.sqrt((1.-alpha_t)*(1.-alpha_t_minus_1_cumprod)/(1.-alpha_t_cumprod))
        else:
            std = 0.
        
        return mean + std*noise


    @torch.no_grad()
    def _reverse_diffusion_clipped(self, xt, t, noise):
        noise_pred = self.model(xt,t)

        alpha_t = self.alphas.gather(-1,t).reshape(xt.shape[0],1,1,1)
        alpha_t_cumprod = self.alphas_cumprod.gather(-1,t).reshape(xt.shape[0],1,1,1)

        pred_clipped = torch.sqrt(1./alpha_t_cumprod)*xt -torch.sqrt(1./alpha_t_cumprod-1.)*noise_pred
        pred_clipped = torch.clamp_(pred_clipped,-1.,1.)


        if t.min() > 0:
            alpha_t_minus_1_cumprod = self.alphas_cumprod.gather(-1,t).reshape(xt.shape[0],1,1,1)
            
            mean = ((1.-alpha_t)*torch.sqrt(alpha_t_minus_1_cumprod)/(1.-alpha_t_cumprod))*pred_clipped + ((1.-alpha_t_minus_1_cumprod)*torch.sqrt(alpha_t)/(1.-alpha_t_cumprod))*xt

            std = torch.sqrt((1.-alpha_t)*(1.-alpha_t_minus_1_cumprod)/(1.-alpha_t_cumprod))
        else:
            mean = (1.-alpha_t)/(1.-alpha_t_cumprod)*pred_clipped
            std = 0.
        
        return mean + std*noise


class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    def __init__(self, model, decay, device):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1-decay) *model_param
        
        super().__init__(model, device, ema_avg, use_buffers = True)