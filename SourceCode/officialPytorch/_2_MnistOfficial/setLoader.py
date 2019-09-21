import torch
from torchvision import datasets, transforms


def setLoader(kwargs, shouldTrain, shouldDownload, batchSize):
    loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=shouldTrain, download=shouldDownload,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batchSize, shuffle=True, **kwargs)
    return loader
