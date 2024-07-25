import copy
import torch

from utils import *
from weight_matching import *



def update_parameters(model, directions, importances):
    niter = len(importances)
    with torch.no_grad():
        for param, direction, importance in tqdm(zip(model.parameters(), directions, importances), desc='updating parameters', total=niter):
            param.data.copy_(param.data + direction * importance)
            # if torch.isinf(param).any():
            #     mask = torch.isinf(param)
            #     print(importance[mask])
            #     print(param[mask])
            #     print(direction[mask])
            #     pass
            # if torch.isinf(importance).any():
            #     pass
            # if torch.isinf(direction).any():
            #     pass
    return model


def update_importances(model_a, model_b, train_loader):
    model_a.train()
    model_b.train()
    device = get_device(model_a)
    loss_fn = torch.nn.CrossEntropyLoss()
    importances_a = []
    importances_b = []
    niter = len(train_loader)
    init = True
    opt_a = torch.optim.Adam(model_a.parameters(), lr=1e-3)
    opt_b = torch.optim.Adam(model_b.parameters(), lr=1e-3)
    for images, labels in tqdm(train_loader, desc='updating importances', total=niter):
        opt_a.zero_grad()
        opt_b.zero_grad()
        images = images.to(device)
        labels = labels.to(device)
        out = model_a(images)
        loss = loss_fn(out, labels)
        loss.backward()
        out = model_b(images)
        loss = loss_fn(out, labels)
        loss.backward()
        for i, param in enumerate(zip(model_a.parameters(), model_b.parameters())):
            param_a, param_b = param
            grad_a = torch.square(param_a.grad)
            grad_b = torch.square(param_b.grad)
            importance_a = (grad_b+1e-7)/(grad_a+grad_b+2e-7)/len(train_loader)
            importance_b = (grad_a+1e-7)/(grad_a+grad_b+2e-7)/len(train_loader)
            # if torch.isinf(importance_a).any():
            #     print('importance a has inf')
            #     print(param_a.grad[torch.isinf(importance_a)])
            #     print(param_b.grad[torch.isinf(importance_a)])
            #     exit(0)
            # if torch.isinf(importance_b).any():
            #     print('importance_b has inf')
            #     print(param_a.grad[torch.isinf(importance_b)])
            #     print(param_b.grad[torch.isinf(importance_b)])
            #     exit(0)
            if init:
                importances_a.append(importance_a)
                importances_b.append(importance_b)
            else:
                importances_a[i] += importance_a
                importances_b[i] += importance_b
        init = False
    return importances_a, importances_b


def simple_avg(model_a, model_b):
    merged_model = copy.deepcopy(model_a)
    with torch.no_grad():
        for m, a, b in zip(merged_model.parameters(), model_a.parameters(), model_b.parameters()):
            m.data.copy_((a.data + b.data)/2)
    return merged_model

def repermute(model_a, model_b):
    # permutation_spec = resnet20_permutation_spec()
    permutation_spec = resnet20_permutation_spec_custom()
    device = get_device(model_a)
    model_a = model_a.cpu()
    model_b = model_b.cpu()
    final_permutation = weight_matching(permutation_spec, model_a.state_dict(), model_b.state_dict())
    updated_params = apply_permutation(permutation_spec, final_permutation, model_b.state_dict())
    model_b.load_state_dict(updated_params)
    model_a = model_a.to(device)
    model_b = model_b.to(device)
    return model_b



def merge(model_a, model_b, train_loader, k):
    directions_from_a_to_b = []
    directions_from_b_to_a = []
    model_b = repermute(model_a, model_b)
    # return simple_avg(model_a, model_b)
    for param_a, param_b in tqdm(zip(model_a.parameters(), model_b.parameters()), desc='initializing directions'):
        directions_from_a_to_b.append((param_b - param_a) / k)
        directions_from_b_to_a.append((param_a - param_b) / k)
    for _ in range(k):
        print(f'-------------------running on {_}/{k} step-------------------')
        importances_a, importances_b = update_importances(model_a, model_b, train_loader)
        model_a = update_parameters(model_a, directions_from_a_to_b, importances_a)
        model_b = update_parameters(model_b, directions_from_b_to_a, importances_b)
    for a, b in zip(model_a.parameters(), model_b.parameters()):
        output = torch.max(torch.abs(a-b))
        print(output)
    exit(0)
    return simple_avg(model_a, model_b)
