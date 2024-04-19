import torchvision
import argparse
import os
import random
import copy
import torch.nn.functional as F
import numpy as np
import torch
from torchvision.models import wide_resnet50_2
import tqdm

from common import *
from utils import *
from mvtec import *
from net import *
from students import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sub_ds  = [
    "bottle"
]
ds = dataset(
    'mvtec',
    'Dataset', subdatasets= sub_ds,
    batch_size=1
)
get_dataloaders = ds[1]
list_of_dataloaders = get_dataloaders()
bottle_loaders = list_of_dataloaders[0]

train_loader = bottle_loaders['train']
test_loader = bottle_loaders['test']

output_folder = 'd_models'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)



def main():
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    backbone = wide_resnet50_2(weights = torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V2)

    extractor = FeatureExtractor(
        backbone,
        layers_to_extract_from=['layer2', 'layer3'],
        device=device,
        input_shape=(3, 224, 224),
    )
    pdn = get_pdn_small(padding = True)

    if not os.path.exists(os.path.join(output_folder, 'channel_mean.pth')):
        channel_mean, channel_std = feature_normalization(extractor, train_loader)
    else:
        channel_mean = torch.load(os.path.join(output_folder, 'channel_mean.pth')).to(device)
        channel_std = torch.load(os.path.join(output_folder, 'channel_std.pth')).to(device)

    pdn.train()
    pdn.to(device)
    optimizer = torch.optim.Adam(pdn.parameters(), lr=1e-4, weight_decay=1e-5)

    state_dict = torch.load('d_models/pdn.pth')
    pdn.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch']

    tqdm_obj = tqdm.tqdm(range(start_epoch, 1000))
    for iter, obj in zip(tqdm_obj, train_loader):
        obj['image'] = obj['image'].to(device)
        obj['mask'] = obj['mask'].to(device)

        target = extractor.embed(obj['image'])[0]
        target = (target - channel_mean) / channel_std
        predict = pdn(obj['image'])
        loss = F.mse_loss(predict, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tqdm_obj.set_postfix({'loss': loss.item()})

        torch.save(
            {'epoch': iter,
            'model': pdn.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': loss.item()},
            os.path.join(output_folder, 'pdn.pth')
        )




@torch.no_grad()
def feature_normalization(extractor, train_loader, steps = 10000):
    mean_outputs = []
    normalization_count = 0
    with tqdm.tqdm(desc='Computing mean of features', total=steps) as pbar:
        for i in train_loader:
            i['image'] = i['image'].to(device)
            output = extractor.embed(i['image'])[0]
            mean_output = torch.mean(output, dim=[0, 2, 3])
            mean_outputs.append(mean_output)
            normalization_count += len(i['image'])
            if normalization_count >= steps:
                pbar.update(steps - pbar.n)
                break
            else:
                pbar.update(len(i['image']))
    channel_mean = torch.mean(torch.stack(mean_outputs), dim=0)
    channel_mean = channel_mean[None, :, None, None]

    mean_distances = []
    normalization_count = 0
    with tqdm.tqdm(desc='Computing variance of features', total=steps) as pbar:
        for i in train_loader:
            i['image'] = i['image'].to(device)
            output = extractor.embed(i['image'])[0]
            distance = (output - channel_mean) ** 2
            mean_distance = torch.mean(distance, dim=[0, 2, 3])
            mean_distances.append(mean_distance)
            normalization_count += len(i['image'])
            if normalization_count >= steps:
                pbar.update(steps - pbar.n)
                break
            else:
                pbar.update(len(i['image']))
    channel_var = torch.mean(torch.stack(mean_distances), dim=0)
    channel_var = channel_var[None, :, None, None]
    channel_std = torch.sqrt(channel_var)

    torch.save(channel_mean, os.path.join(output_folder, 'channel_mean.pth'))
    torch.save(channel_std, os.path.join(output_folder, 'channel_std.pth'))

    return channel_mean, channel_std

if __name__ == '__main__':
    main()