from torch.optim import SGD
from .anytime_sgd import AnyTimeSGD
from .mu2sgd import Mu2SGD
from .storm import STORM


OPTIMIZER_REGISTRY = {
    'sgd': SGD,
    'momentum': SGD,
    'anytime_sgd': AnyTimeSGD,
    'mu2sgd': Mu2SGD,
    'storm': STORM
}


