import re
import spacy
import torch
from torch.autograd import Variable


NLP = spacy.load('en')


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def detach(x):
    """Wraps hidden states in new variables, to detach them from their history."""
    if type(x) == Variable:
        return Variable(x.data)
    else:
        return tuple(detach(v) for v in x)
    
    
def tokenizer(text):
    text = re.sub(b'\u200c'.decode("utf-8", "strict"), " ", text)   # replace half-spaces with spaces
    text = re.sub('\n', ' ', text)
    text = re.sub('-', ' - ', text)
    text = re.sub('[ ]+', ' ', text)
    text = re.sub('\.', ' .', text)
    text = re.sub('\،', ' ،', text)
    text = re.sub('\؛', ' ؛', text)
    text = re.sub('\؟', ' ؟', text)
    text = re.sub('\. \. \.', '...', text)
    
    return [w.text for w in NLP.tokenizer(str(text))]