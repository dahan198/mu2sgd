from .cifar10 import CIFAR10
from .mnist import MNIST


DATASET_REGISTRY = {
    'mnist': MNIST,
    'cifar10': CIFAR10
}