import os
import torchvision as tv
from torchvision import transforms


# def get_dataset_info(name):
#     if name == 'cifar10':
#         return {'num_classes': 10}
#     elif name == 'cifar100':
#         return {'num_classes': 100}
#     else:
#         raise ValueError(f'{name} is not found')


def get_dataset(name, train, my_path=None, imagenet_path=None, transform=transforms.ToTensor()):
    name = name.lower()
    cls = getattr(tv.datasets, name.upper())
    if my_path is None:
        path = os.path.join('data', f'{name}')
    else:
        path = os.path.join(f'{my_path}/data', f'{name}')
    # transform = transforms.ToTensor()

    if name in ['mnist', 'cifar10', 'cifar100']:
        train_split = True if train else False
        # download if fails
        try:
            dataset = cls(
                path, train=train_split,
                download=False,
                transform=transform)
        except RuntimeError:
            dataset = cls(
                path,
                train=train_split,
                download=True,
                transform=transform)
    elif name == 'imagenet':
        dataset = tv.datasets.ImageFolder(imagenet_path, transform=transform)
    else:
        train_split = 'train' if train else 'test'
        # download if fails
        try:
            dataset = cls(
                path,
                split=train_split,
                download=False,
                transform=transform)
        except RuntimeError:
            dataset = cls(
                path,
                split=train_split,
                download=True,
                transform=transform)
    return dataset
