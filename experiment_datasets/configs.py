
import torchvision

cifar50 = {
    # 'dir': './data/cifar-100-python',
    'dir': '/nfs/py/data',
    'num_classes': 100,
    'wrapper': torchvision.datasets.CIFAR100,
    'batch_size': 500,
    'type': 'cifar',
    'shuffle_train': True,
    'shuffle_test': False,
    'num_workers': 2,
}