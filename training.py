import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import logging

from UNet import UNet
from diffusion import Diffusion
from utils import *
from students import *

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pdn = get_pdn_small(padding = True)
model = UNet(c_in=1536, c_out=1536, time_dim=28, device = device)
diffusion = Diffusion(device=device, input_size=28)


ds = dataset(
    'mvtec',
    'Dataset', subdatasets= ['bottle'],
    batch_size=1
)
bottle_loaders = ds[1]()[0]
train_loader = bottle_loaders['train']
test_loader = bottle_loaders['test']

output_folder  = 'disnet'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

epochs = 100

pdn.load_state_dict(torch.load('d_models/pdn.pth')['model'])
pdn.to(device)
model.to(device)
pdn.eval()
model.train()

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
logger = SummaryWriter(log_dir='logs', flush_secs=10, filename_suffix='disnet')
logging.basicConfig(filename='logs/disnet.log', level=logging.DEBUG)

for epoch in range(epochs):
    logging.info("Epoch: %d", epoch)
    pbar = tqdm.tqdm(train_loader)
    for i, sample in enumerate(pbar):
        image = sample['image'].to(device)
        features = pdn(image)
        t = diffusion.sample_timesteps(features.shape[0])
        x_t, noise = diffusion.noise_input(features, t)
        predicted_noise = model(x_t, t)
        loss = F.mse_loss(predicted_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description(f"Loss: {loss.item()}")
        logger.add_scalar('Loss', loss.item(), epoch * len(train_loader) + i)
    torch.save(model.state_dict(), f'{output_folder}/disnet.pth')



