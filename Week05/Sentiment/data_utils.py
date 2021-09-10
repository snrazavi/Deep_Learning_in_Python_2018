import os
import re
import spacy
import pickle
from glob import glob
from tqdm import tqdm

import torch
from torch.utils.data import Dataset


NLP = spacy.load('en_core_web_sm')


def tokenizer(text):
    text = re.sub(r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’;]", " ", str(text))
    text = re.sub(r"[ ]+", " ", text)
    text = re.sub(r"\!+", "!", text)
    text = re.sub(r"\,+", ",", text)
    text = re.sub(r"\?+", "?", text)
    return [x.text for x in NLP.tokenizer(text) if x.text != " "]


class Vocabulary(object):
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.count = 0
    
    def add_word(self, word):
        if not word in self.word2index:
            self.word2index[word] = self.count
            self.word2count[word] = 1
            self.index2word[self.count] = word
            self.count += 1
        else:
            self.word2count[word] += 1
    
    def add_sentence(self, sentence):
        for word in self.tokenizer(sentence):
            self.add_word(word)
            
    def __len__(self):
        return self.count

    
if __name__ == '__main__':
    pass
