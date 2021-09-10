from torch.autograd import Variable


def detach(x):
    """ Detach hidden states from their history."""
    return Variable(x.data) if type(x) == Variable else tuple(detach(v) for v in x)