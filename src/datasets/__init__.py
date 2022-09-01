from .get_dataset import get_dataset
from .get_transform import get_transform


dataset_info = {
    'mnist': {'num_classes': 10},
    'cifar10': {'num_classes': 10},
    'cifar100': {'num_classes': 100},
    'imagenet': {'num_classes': 1000},
    'hamming': {'num_classes': 21},
}