from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch
import torchvision.datasets
import torchvision.models
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from sklearn.metrics import accuracy_score
import datetime
import os
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_data_loaders(train_data: Dataset, test_data: Dataset, batch_size:int):
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_data_loader, test_data_loader

def test_model(model: nn.Module, data_loader: DataLoader, criterion):
    labels = []
    predictions = []
    model.eval()
    test_loss = 0
    tot_cnt = 0
    correct_cnt = 0
    with torch.no_grad():
        for data in data_loader:
            image = data[0].to(device)
            label = data[1].to(device)
            pred = model(image)
            pred_class = torch.argmax(pred, dim=1)
            loss = criterion(pred, label)
            test_loss += loss.sum()

            tot_cnt += pred_class.size(0)
            correct_cnt += torch.eq(pred_class, label).float().sum().item()

            for l, p in zip(label, pred_class):
                labels.append(l.item())
                predictions.append(p.item())

    acc = correct_cnt / tot_cnt
    return accuracy_score(labels, predictions), test_loss / len(data_loader)

def train_model(model: nn.Module, train_data_loader: DataLoader, test_data_loader: DataLoader, optimizer, criterion, model_path, num_epochs=50, model_weights=None):
    """
    Parameters:
    model - model to be trained \n
    data_loader - data loader containing the training data \n
    criterion - evaluation function to be used \n
    optimizer - optimizer to optimize the weights \n
    """
    dsa_param = ParamDiffAug()
    dsa_strategy = 'color_crop_cutout_flip_scale_rotate'

    if model_weights:
        model.load_state_dict(model_weights)
    model.to(device)

    start_time = datetime.datetime.now()
    best_acc = 0
    train_record = []
    test_record = []
    scheduler = ReduceLROnPlateau(optimizer, patience=1, mode='max', threshold=0.001, factor=0.5)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        loop = tqdm(train_data_loader)
        for data in loop:
            optimizer.zero_grad()
            image = data[0].to(device)
            label = data[1].to(device)
            image = DiffAugment(image, dsa_strategy, param=dsa_param)

            pred = model(image)
            loss = criterion(pred, label)
            loss.backward()

            with torch.no_grad():
                train_loss += loss.item()

            loop.set_postfix({"loss": loss.item()})
            loop.set_description(f"Epoch {epoch+1}")

            optimizer.step()

        avg_train_loss = train_loss / len(train_data_loader)
        print(f"epoch: {epoch+1} done, loss: {avg_train_loss}")
        acc, test_loss = test_model(model, test_data_loader, criterion)
        print(f"Acc: {acc}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f"{model_path}/model.bin")
        train_record.append(train_loss)
        test_record.append(test_loss)
        scheduler.step(test_loss)
        # print(scheduler.get_last_lr())
        train_loss = 0
        
    end_time = datetime.datetime.now()
    print(f"Time taken: {(end_time - start_time).total_seconds()} seconds")
    return train_record, test_record

def train(model, model_path, train_data_loader, test_data_loader, num_epochs, optimizer='sgd', lr=1e-2):
    assert optimizer in ['adam', 'sgd'], 'Not valid option'
    criterion = nn.CrossEntropyLoss()
    if optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)  #1e-4 to converge + Adam
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005) 
    
    return train_model(model, train_data_loader, test_data_loader, optimizer, criterion, model_path, num_epochs=num_epochs)
    
def test(model, model_path, test_data_loader, criterion):
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.to(device)
    return test_model(model, test_data_loader, criterion)



def get_daparam(dataset, model, model_eval, ipc):
    # We find that augmentation doesn't always benefit the performance.
    # So we do augmentation for some of the settings.

    dc_aug_param = dict()
    dc_aug_param['crop'] = 4
    dc_aug_param['scale'] = 0.2
    dc_aug_param['rotate'] = 45
    dc_aug_param['noise'] = 0.001
    dc_aug_param['strategy'] = 'none'

    if dataset == 'MNIST':
        dc_aug_param['strategy'] = 'crop_scale_rotate'

    if model_eval in ['ConvNetBN']:  # Data augmentation makes model training with Batch Norm layer easier.
        dc_aug_param['strategy'] = 'crop_noise'

    return dc_aug_param

class ParamDiffAug():
    def __init__(self):
        self.aug_mode = 'S' #'multiple or single'
        self.prob_flip = 0.5
        self.ratio_scale = 1.2
        self.ratio_rotate = 15.0
        self.ratio_crop_pad = 0.125
        self.ratio_cutout = 0.5 # the size would be 0.5x0.5
        self.ratio_noise = 0.05
        self.brightness = 1.0
        self.saturation = 2.0
        self.contrast = 0.5

def set_seed_DiffAug(param):
    if param.latestseed == -1:
        return
    else:
        torch.random.manual_seed(param.latestseed)
        param.latestseed += 1

def rand_scale(x, param):
    # x>1, max scale
    # sx, sy: (0, +oo), 1: orignial size, 0.5: enlarge 2 times
    ratio = param.ratio_scale
    set_seed_DiffAug(param)
    sx = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    set_seed_DiffAug(param)
    sy = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    theta = [[[sx[i], 0,  0],
            [0,  sy[i], 0],] for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.batchmode: # batch-wise:
        theta[:] = theta[0]
    grid = F.affine_grid(theta, x.shape, align_corners=True).to(x.device)
    x = F.grid_sample(x, grid, align_corners=True)
    return x


def rand_rotate(x, param): # [-180, 180], 90: anticlockwise 90 degree
    ratio = param.ratio_rotate
    set_seed_DiffAug(param)
    theta = (torch.rand(x.shape[0]) - 0.5) * 2 * ratio / 180 * float(np.pi)
    theta = [[[torch.cos(theta[i]), torch.sin(-theta[i]), 0],
        [torch.sin(theta[i]), torch.cos(theta[i]),  0],]  for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.batchmode: # batch-wise:
        theta[:] = theta[0]
    grid = F.affine_grid(theta, x.shape, align_corners=True).to(x.device)
    x = F.grid_sample(x, grid, align_corners=True)
    return x


def rand_flip(x, param):
    prob = param.prob_flip
    set_seed_DiffAug(param)
    randf = torch.rand(x.size(0), 1, 1, 1, device=x.device)
    if param.batchmode: # batch-wise:
        randf[:] = randf[0]
    return torch.where(randf < prob, x.flip(3), x)


def rand_brightness(x, param):
    ratio = param.brightness
    set_seed_DiffAug(param)
    randb = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.batchmode:  # batch-wise:
        randb[:] = randb[0]
    x = x + (randb - 0.5)*ratio
    return x


def rand_saturation(x, param):
    ratio = param.saturation
    x_mean = x.mean(dim=1, keepdim=True)
    set_seed_DiffAug(param)
    rands = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.batchmode:  # batch-wise:
        rands[:] = rands[0]
    x = (x - x_mean) * (rands * ratio) + x_mean
    return x


def rand_contrast(x, param):
    ratio = param.contrast
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    set_seed_DiffAug(param)
    randc = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.batchmode:  # batch-wise:
        randc[:] = randc[0]
    x = (x - x_mean) * (randc + ratio) + x_mean
    return x


def rand_crop(x, param):
    # The image is padded on its surrounding and then cropped.
    ratio = param.ratio_crop_pad
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    if param.batchmode:  # batch-wise:
        translation_x[:] = translation_x[0]
        translation_y[:] = translation_y[0]
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


def rand_cutout(x, param):
    ratio = param.ratio_cutout
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    if param.batchmode:  # batch-wise:
        offset_x[:] = offset_x[0]
        offset_y[:] = offset_y[0]
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'crop': [rand_crop],
    'cutout': [rand_cutout],
    'flip': [rand_flip],
    'scale': [rand_scale],
    'rotate': [rand_rotate],
}

def DiffAugment(x, strategy='', seed = -1, param = None):
    if seed == -1:
        param.batchmode = False
    else:
        param.batchmode = True

    param.latestseed = seed

    if strategy == 'None' or strategy == 'none':
        return x

    if strategy:
        if param.aug_mode == 'M': # original
            for p in strategy.split('_'):
                for f in AUGMENT_FNS[p]:
                    x = f(x, param)
        elif param.aug_mode == 'S':
            pbties = strategy.split('_')
            set_seed_DiffAug(param)
            p = pbties[torch.randint(0, len(pbties), size=(1,)).item()]
            for f in AUGMENT_FNS[p]:
                x = f(x, param)
        else:
            exit('Error ZH: unknown augmentation mode.')
        x = x.contiguous()
    return x

"""
    dsa augmentation during training
"""
