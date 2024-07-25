import os
import sys

import torch
import torch.nn as nn

from copy import deepcopy
from torch.amp import autocast
from tqdm import tqdm


def is_valid_pair(model_dir, pair, model_type):
    paths = os.listdir(os.path.join(model_dir, pair[0]))
    flag = True
    for path in paths:
        if f'{model_type}_v0.pth.tar' not in path:
            flag = False
    return flag


def find_pairs(str_splits):
    pairs = []
    for i, str_split_i in enumerate(str_splits):
        try:
            split_i = set([int(k) for k in str_split_i.split('_')])
        except:
            continue
        for str_split_j in str_splits[i + 1:]:
            try:
                split_j = set([int(k) for k in str_split_j.split('_')])
            except:
                continue
            if len(split_i.intersection(split_j)) == 0:
                pairs.append((str_split_i, str_split_j))
    return pairs


def find_runable_pairs(model_dir, model_name, skip_pair_idxs=[]):
    run_pairs = []
    valid_pairs = [pair for pair in find_pairs(os.listdir(model_dir)) if is_valid_pair(model_dir, pair, model_name)]
    for idx, pair in enumerate(valid_pairs):
        if idx in skip_pair_idxs:
            continue
        run_pairs += [pair]
    return run_pairs


def split_str_to_ints(split):
    return [int(i) for i in split.split('_')]

def inject_pair(config, pair, ignore_bases=False):
    model_name = config['model']['name']
    config['dataset']['class_splits'] = [split_str_to_ints(split) for split in pair]
    if not ignore_bases:
        config['model']['bases'] = [os.path.join(config['model']['dir'], split, f'{model_name}_v0.pth.tar') for split in pair]
    return config

def get_device(model):
    """Get the device of the model."""
    return next(iter(model.parameters())).device


# use the train loader with data augmentation as this gives better results
# taken from https://github.com/KellerJordan/REPAIR
def reset_bn_stats(model, loader, reset=True):
    """Reset batch norm stats if nn.BatchNorm2d present in the model."""
    device = get_device(model)
    has_bn = False
    # resetting stats to baseline first as below is necessary for stability
    for m in model.modules():
        if type(m) == nn.BatchNorm2d:
            if reset:
                m.momentum = None  # use simple average
                m.reset_running_stats()
            has_bn = True

    if not has_bn:
        return model

    # run a single train epoch with augmentations to recalc stats
    model.train()
    with torch.no_grad(), autocast('cuda'):
        for images, _ in tqdm(loader, desc='Resetting batch norm'):
            _ = model(images.to(device))
    return model


def prepare_resnets(config, device):
    """ Load all pretrained resnet models in config. """
    bases = []
    name = config['name']

    if 'x' in name:
        width = int(name.split('x')[-1])
        name = name.split('x')[0]
    else:
        width = 1

    if 'resnet20' in name:
        from models.resnet20 import resnet20 as wrapper_w
        wrapper = lambda num_classes: wrapper_w(width, num_classes)
    elif 'resnet50' in name:
        from torchvision.models import resnet50 as wrapper
    elif 'resnet18' in name:
        from torchvision.models import resnet18 as wrapper
    else:
        raise NotImplementedError(config['name'])

    output_dim = config['output_dim']
    for base_path in tqdm(config['bases'], desc="Preparing Models"):
        base_sd = torch.load(base_path, map_location=torch.device(device))

        # Remove module for dataparallel
        for k in list(base_sd.keys()):
            if k.startswith('module.'):
                base_sd[k.replace('module.', '')] = base_sd[k]
                del base_sd[k]

        base_model = wrapper(num_classes=output_dim).to(device)
        base_model.load_state_dict(base_sd)
        bases.append(base_model)
    new_model = wrapper(num_classes=output_dim).to(device)
    return {
        'bases': bases,
        'new': new_model  # this will be the merged model
    }


def prepare_data(config, device='cuda'):
    """ Load all dataloaders required for experiment. """
    if isinstance(config, list):
        return [prepare_data(c, device) for c in config]

    dataset_name = config['name']

    import experiment_datasets.configs as config_module
    data_config = deepcopy(getattr(config_module, dataset_name))
    data_config.update(config)
    data_config['device'] = device

    if data_config['type'] == 'cifar':
        from experiment_datasets.cifar import prepare_train_loaders, prepare_test_loaders
        train_loaders = prepare_train_loaders(data_config)
        test_loaders = prepare_test_loaders(data_config)
    else:
        raise NotImplementedError(config['type'])

    return {
        'train': train_loaders,
        'test': test_loaders
    }


def prepare_models(config, device='cuda'):
    """ Load all pretrained models in config. """
    if config['name'].startswith('resnet'):
        return prepare_resnets(config, device)

