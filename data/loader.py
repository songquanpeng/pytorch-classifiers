import numpy as np
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10


def _make_balanced_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, len(weights))


def get_train_loader(archive_dir, img_size, batch_size, dataset, num_workers=4, **kwargs):
    transform_list = []
    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
    if kwargs["img_dim"] == 1:
        transform_list.append(transforms.Grayscale())
        norm_mean = [0.5]
        norm_std = [0.5]

    if dataset == 'MNIST':
        transform_list.extend([
            transforms.Resize([img_size, img_size]),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])
        transform = transforms.Compose(transform_list)
        dataset = MNIST(archive_dir, train=True, download=True, transform=transform)
    elif dataset == 'CIFAR-10':
        transform_list.extend([
            transforms.Resize([img_size, img_size]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])
        transform = transforms.Compose(transform_list)
        dataset = CIFAR10(archive_dir, train=True, download=True, transform=transform)
    else:
        raise NotImplementedError

    sampler = _make_balanced_sampler(dataset.targets)

    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           sampler=sampler,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=True)


def get_test_loader(archive_dir, img_size, batch_size, dataset=None, num_workers=4, **kwargs):
    transform_list = []
    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
    if kwargs["img_dim"] == 1:
        transform_list.append(transforms.Grayscale())
        norm_mean = [0.5]
        norm_std = [0.5]

    transform_list.extend([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])
    transform = transforms.Compose(transform_list)

    if dataset == 'MNIST':
        dataset = MNIST(archive_dir, train=False, download=False, transform=transform)
    elif dataset == 'CIFAR-10':
        dataset = CIFAR10(archive_dir, train=False, download=False, transform=transform)
    else:
        raise NotImplementedError

    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=num_workers,
                           pin_memory=True)
