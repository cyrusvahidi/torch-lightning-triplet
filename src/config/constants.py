from enum import Enum

class ModelType(Enum):
    MNIST = 1
    Triplet = 2

class LossType(Enum):
    ONLINE_BATCH_ALL = 1


MODEL_DICT = {
    'mnist': ModelType.MNIST,
    'triplet': ModelType.Triplet
}

LOSS_DICT = {
    'online_batch_all': LossType.ONLINE_BATCH_ALL
}