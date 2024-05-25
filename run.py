# TO DO:
# We want to run this file to either generate from already saved model weights or train given some dataset.
import yaml
import tqdm
import argparse
from utils import load_dataset, training_steps, evaluation_step
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models import *
import datetime
import json
import torchvision
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

models_dict = {'VAE_1D':VAE_1D, 'VAE_3D':VAE_3D, 'VAE_complex':VAE_complex, 'MultiStage_VAE': MultiStage_VAE, 'ResNet': ResNet, 'VQ_VAE': VQ_VAE, 'PixelCNN': PixelCNN}

parser = argparse.ArgumentParser()

parser.add_argument('--config','-c', dest = 'fileconfig' , metavar = 'FILENAME',help = "Config filepath", default = './parameters/VAE_parameters.yaml')
parser.add_argument('--weights', '-w', dest = 'path_weights', help = "Name of weights", default = None)
parser.add_argument('--mode', '-m',dest = 'mod', choices = ['sample','train', 'test'],default = 'train',help = "Option to sample using already computed weights")
parser.add_argument('--save', dest = 'save_output', action = 'store_true' )
args = parser.parse_args()

with open(args.fileconfig, 'rb') as f:
    try:
        config = yaml.safe_load(f)
    except yaml.YAMLError as exception:
        print(exception)
name_experiment = config['model']['name']+'_' + config['dataset']['name']
print(name_experiment)

data, unnormalize, nb_classes = load_dataset(config)
dataloader = DataLoader(data, batch_size = config['dataset']['batch_size'], shuffle = True)

model = models_dict[config['model']['name']](config, img_shape = data[0][0].shape,  nb_classes=nb_classes).to(device)
if args.path_weights is not None:
    model.load_state_dict(torch.load('./model_saves/' + args.path_weights+'.pth'))
    print('Weights loaded!')
    if args.mod == 'train':
        name_experiment += '_improved'

classification = True if config['model']['name']=='ResNet' else False
log_dir = './runs/'+name_experiment+'/' +datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir)

writer.add_text('Config file', json.dumps(config))
if args.mod != 'sample':
    try:
        kl_weight = config['optimization']['kl_weight']
    except:
            kl_weight = 1

    if args.mod == 'train':      
        losses = training_steps(model, dataloader,unnormalize=unnormalize,classification= classification, logger = writer)
        torch.save(model.state_dict(), log_dir+'/'+name_experiment+'.pth')
        
        plt.figure(0)
        plt.plot(losses)

    test_data, unnormalize, nb_classes = load_dataset(config, training=False)
    test_dataloader = DataLoader(test_data, batch_size = config['dataset']['batch_size'], shuffle = True)
    evaluation_step(model, test_dataloader, unnormalize, classification, writer)

    if not classification:
        img_sample = unnormalize(model.sample(nb_images = 16))
        sample_grid = torchvision.utils.make_grid(img_sample, nrow = 4)
        writer.add_image('Sampled image', sample_grid)
        if args.save_output:
            torchvision.utils.save_image(sample_grid, './output_images/'+name_experiment+'.png')
        if model.latent_dim == 2:
            manifold = model.eval_manifold(unnormalize)
            writer.add_image('Manifold', manifold)
            if args.save_output:
                torchvision.utils.save_image(manifold, './output_images/'+name_experiment+'_manifold.png')
writer.close()