from .resnet import *
from .simple_conv import SimpleConv
from .logistic_regression import LogisticRegression

MODEL_REGISTRY = {
    'resnet18': ResNet18,
    'simple_conv': SimpleConv,
    'logistic_regression': LogisticRegression,
}
