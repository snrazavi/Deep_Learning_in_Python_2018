import torch
from torch.autograd import Variable


def to_var(x, volatile=False):
    x = x.cuda() if torch.cuda.is_available() else x
    return Variable(x, volatile=volatile)


def detach(x):
    """ Detach hidden states from their history."""
    return Variable(x.data) if type(x) == Variable else tuple(detach(v) for v in x)