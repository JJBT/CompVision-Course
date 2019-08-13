import torchvision.datasets as datasets
import torch


def get_cifar():
    CIFAR_train = datasets.CIFAR10('./', download=True, train=True)
    CIFAR_test = datasets.CIFAR10('./', download=True, train=False)

    X_train = torch.FloatTensor(CIFAR_train.data)
    y_train = torch.LongTensor(CIFAR_train.targets)
    X_test = torch.FloatTensor(CIFAR_test.data)
    y_test = torch.LongTensor(CIFAR_test.targets)

    X_train /= 255
    X_test /= 255

    X_train = X_train.permute(0, 3, 1, 2)
    X_test = X_test.permute(0, 3, 1, 2)

    return X_train, y_train, X_test, y_test
