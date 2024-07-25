import argparse
import copy
import time
import torch

from approach import *
from tqdm import tqdm
from utils import *

def test(model, test_loader):
    model.eval()
    device = get_device(model)
    correct = 0
    total = 0
    for images, labels in tqdm(test_loader, desc='Testing'):
        images = images.to(device)
        labels = labels.to(device)
        total += images.size(0)
        output = model(images)
        predicted = torch.argmax(output, dim=1)
        correct += (predicted == labels).sum().item()
    print(f'accuracy: {100 * correct / total}%')

def main(config):
    model_a = config['models']['bases'][0]
    model_b = config['models']['bases'][1]
    train_loader = config['data']['train']['full']
    test_loader = config['data']['test']['full']
    # merged_model = merge(model_a, model_b, train_loader, config['k'])
    merged_model = simple_avg(model_a, model_b)
    reset_bn_stats(merged_model, train_loader)
    # torch.save(merged_model.state_dict(), './checkpoints/approach_merged.pth')
    test(merged_model, test_loader)

def init(parser):
    args = parser.parse_args()
    config = copy.deepcopy(getattr(__import__('configs.' + args.config_file), args.config_file).config)
    pair = find_runable_pairs(config['model']['dir'], config['model']['name'])[0]
    config = inject_pair(config, pair)
    config['device'] = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    data = prepare_data(config['dataset'], device=config['device'])
    config['data'] = data
    config['model']['output_dim'] = len(data['test']['class_names'])
    config['models']=prepare_models(config['model'], device=config['device'])
    train_loader = config['data']['train']['full']
    test_loader = config['data']['test']['full']
    # config['models']['bases'] = [reset_bn_stats(base_model, train_loader) for base_model in config['models']['bases']]
    new_config = {
        'k': args.k,
    }
    config.update(new_config)
    return config


if __name__ == '__main__':
    start_time = time.time()  # 记录开始时间


    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', required=False, type=str, default='resnet20_cifar50')
    parser.add_argument('--k', required=False, type=int, default=20)
    parser.add_argument('--checkpoint', required=False, type=str, default='./checkpoints/approch_merged.pth')

    # True False
    debugging = False
    config = init(parser)
    main(config)



    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算消耗的时间
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"time cost: {int(hours)} hours:{int(minutes)} min:{int(seconds)} sec")
