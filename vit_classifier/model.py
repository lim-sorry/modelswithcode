import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(type(self), self).__init__(*args, **kwargs)
        nn.Parameter()