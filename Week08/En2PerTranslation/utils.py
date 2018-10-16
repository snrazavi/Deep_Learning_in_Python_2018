import time
import math
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import arabic_reshaper as ar
from bidi.algorithm import get_display


# a handy helper function in torch
def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def to_np(t):
    if isinstance(t, Variable):
        t = t.data 
    return t.cpu().numpy() if torch.cuda.is_available else t.numpy()

def to_persian_text(text):
    """This is used in plotting"""
    return get_display(ar.reshape(text))


### Helper functions to work with time.

def to_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%02dm %02ds' % (m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (to_minutes(s), to_minutes(rs))


def indexes_from_sentence(vocab, sentence):
    return [vocab.wtoi(word) for word in sentence.split(' ')]

def variable_from_sentence(vocab, sentence, volatile=False):
    indexes = indexes_from_sentence(vocab, sentence)
    indexes.append(3)  # 3 is EOS token
    return to_var(torch.LongTensor(indexes).view(-1, 1), volatile=volatile)


### Plotting

def show_plot(losses):
    trn_losses, val_losses = zip(*losses)
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(trn_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.show()


def show_attention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)                
    cax = ax.matshow(attentions, cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' '), rotation=90)
    output_words = to_persian_text(' '.join(reversed(output_words))).split()
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
