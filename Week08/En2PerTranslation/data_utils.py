import torch
import pandas as pd
import numpy as np


def filter_pair(pair, MAX_LENGTH):
    src, tgt = pair
    return len(src.split(' ')) < MAX_LENGTH and len(tgt.split(' ')) < MAX_LENGTH

def filter_pairs(pairs, MAX_LENGTH):
    return [pair for pair in pairs if filter_pair(pair, MAX_LENGTH)]

def load_corpus(data_dir, lang1="en", lang2="fa", min_count=3, max_vocab=30000, reverse=False):
    corpus_filename = f'{data_dir}/{lang1}-{lang2}-tok.csv'
    corpus_csv = pd.read_csv(corpus_filename, sep='\t', header=None)
    sentence_pairs = list(zip(corpus_csv[0], corpus_csv[1]))

    if reverse:
        sentence_pairs = list(reversed(pair) for pair in sentence_pairs)
    
    return sentence_pairs

def split(sentence_pairs, split_ratio=0.2, seed=1234):
    np.random.seed(seed)
    N = int(split_ratio * len(sentence_pairs))
    sentences = sentence_pairs[:]
    np.random.shuffle(sentences)
    val_sentence_pairs = sentences[:N]
    trn_sentence_pairs = sentences[N:]
    return trn_sentence_pairs, val_sentence_pairs


def collate_fn(sentence_pairs_ids):
    """Meage a list of samples to create a minibatch"""    
    bs = len(sentence_pairs_ids)
    sentence_pairs_ids.sort(key=lambda x: len(x[0]), reverse=True)
    src_sents_ids, tgt_sents_ids = zip(*sentence_pairs_ids)
    src_max_len = max([len(ids) for ids in src_sents_ids])
    tgt_max_len = max([len(ids) for ids in tgt_sents_ids])
    lengths = [len(ids) for ids in src_sents_ids]
    
    X = torch.zeros(bs, src_max_len).long()
    for i, ids in enumerate(src_sents_ids):
        X[i, :len(ids)] = torch.Tensor(ids).long()
        
    Y = torch.zeros(bs, tgt_max_len).long()
    for i, ids in enumerate(tgt_sents_ids):
        Y[i, :len(ids)] = torch.Tensor(ids).long()
                
    assert X.size(1) == max(lengths)
    return X.t(), Y.t(), lengths