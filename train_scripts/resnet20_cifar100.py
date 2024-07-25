import argparse
import copy
import itertools
import time
from collections import OrderedDict
import gc
import numpy as np
import torchvision
import yaml
import math
import torch
import torch.nn as nn
from tqdm import tqdm
from models.resnet20 import resnet20
from torch.cuda.amp import autocast

import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def train(model, train_loader, test_loader, optimizer, scheduler=None, config=None):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    best_acc = 0
    # for epoch in range(opt['num_epochs']):
    for epoch in range(config['epochs']):
        # running_loss = 0
        # for t, loader in zip(['piece1', 'piece2'], train_loader):
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config["epochs"]}'):
            images = images.to(device)
            labels = labels.to(device)

            out = model(images)
            loss = loss_fn(out, labels)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        acc = test(model, test_loader)
        if config['debug'] is False:
            # writer.add_scalar(f'{tp}/acc', acc, epoch+(warm if tp=='P' else 0))
            config['writer'].add_scalar(f'{config["epochs"]} epochs/acc', acc, epoch)
            if 'warm' in config:
                config['writer'].add_scalar(f'{config["epochs"] + config["warm"]} epochs/acc', acc, epoch + config['warm'])
            config['writer'].flush()
        if acc>best_acc:
            best_acc = acc
            torch.save(model, f'./checkpoints/cifar100.pth')
    return best_acc


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for val_images, labels_val in tqdm(test_loader, desc=f'testing {len(test_loader)} samples'):
            total += val_images.size(0)
            val_images = val_images.to(device)
            labels_val = labels_val.to(device)
            out = model(val_images)
            predicted = torch.argmax(out, dim=1)
            correct += (predicted == labels_val).sum().item()

            # the losses are averaged within the MTL optimizers, possibly after manipulations per datapoint
            loss = loss_fn(out, labels_val)
            test_loss += loss.item()

    acc = correct / total * 100
    # print(f'loss of test: {test_loss/len(test_loader)}')
    print(f'acc of permutation model: {acc}%')
    return acc


def reset_perm_bn_stats(model, loader, reset=True):
    """Reset batch norm stats if nn.BatchNorm2d present in the model."""

    # run a single train epoch with augmentations to recalc stats
    for m in model:
        model[m].train()
    with torch.no_grad(), autocast():
        for images, _ in tqdm(loader, desc='Resetting batch norm'):
            _ = model['perm'](images.to(device), model_a=model['model_a'], model_b=model['model_b'])
    return model


def init(arguments):
    if arguments.debug == 'true':
        debug = True
    else:
        debug = False
    # text_dict={'x': 250, 'lr':1e-8}
    finetune_config = {'epochs': arguments.epochs, 'debug': debug, 'batch_size': arguments.batch_size}
    if debug is False:
        t = time.localtime()
        writer = SummaryWriter(f'./logs/cifar100/{t.tm_mon:02d}-{t.tm_mday:02d}-{t.tm_hour:02d}-{t.tm_min:02d}')
        writer.add_text('args', str(finetune_config))
        finetune_config['writer'] = writer
        # writer.add_text('args', str(opt))
    elif debug is True:
        t = time.localtime()
        print(f'debugging at {t.tm_mon:02d}-{t.tm_mday:02d}-{t.tm_hour:02d}-{t.tm_min:02d}')
        pass
    return finetune_config


if __name__ == '__main__':
    start_time = time.time()  # 记录开始时间
    use_dataset = 'cifar100'
    parser = argparse.ArgumentParser(description='Process with checkpoint')
    parser.add_argument('--seed', required=False, type=int, default=0)
    parser.add_argument('--debug', required=False, type=str, default='true')
    parser.add_argument('--epochs', required=False, type=int, default=10)
    parser.add_argument('--batch_size', required=False, type=int, default=500)
    arguments = parser.parse_args()
    # 237
    random_seed = arguments.seed
    # random_seed = 237
    print(f'seed: {random_seed}')
    if random_seed > 0:
        torch.manual_seed(random_seed)
    config = init(arguments)

    # prepare zip model and dataloaders
    CIFAR_MEAN = [125.307, 122.961, 113.8575]
    CIFAR_STD = [51.5865, 50.847, 51.255]
    normalize = T.Normalize(np.array(CIFAR_MEAN) / 255, np.array(CIFAR_STD) / 255)
    denormalize = T.Normalize(-np.array(CIFAR_MEAN) / np.array(CIFAR_STD), 255 / np.array(CIFAR_STD))
    train_transform = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(32, padding=4), T.ToTensor(), normalize])
    test_transform = T.Compose([T.ToTensor(), normalize])
    train_dataset = torchvision.datasets.CIFAR100(root='/nfs/py/data', train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR100(root='/nfs/py/data', train=False, download=True, transform=test_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True)



    # train
    ne_iters = len(train_loader)
    model = resnet20(w=16, num_classes=100).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=1e-7, total_iters=ne_iters)
    best_acc = train(model=model, train_loader=train_loader, test_loader=test_loader, optimizer=optimizer,
                            scheduler=scheduler, config=config)
    print(f'best acc: {best_acc}')

    end_time = time.time()  # 记录结束时间

    elapsed_time = end_time - start_time  # 计算消耗的时间

    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)

    print(f"总耗时{int(hours)}小时{int(minutes)}分钟{int(seconds)}秒")
