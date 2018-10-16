#import nltk
import re
import pickle
import argparse
from collections import Counter
from tqdm import tqdm_notebook
from coco import COCO

import spacy
NLP = spacy.load('en')


def tokenizer(text):
    text = re.sub(b'\u200c'.decode("utf-8", "strict"), " ", text)   # replace half-spaces with spaces
    text = re.sub('\n', ' ', text)
    text = re.sub('-', ' - ', text)
    text = re.sub('[ ]+', ' ', text)
    text = re.sub('\.', ' .', text)
    text = re.sub('\طŒ', ' طŒ', text)
    text = re.sub('\ط›', ' ط›', text)
    text = re.sub('\طں', ' طں', text)
    text = re.sub('\. \. \.', '...', text)
    
    return [w.text for w in NLP.tokenizer(str(text))]
    
    
class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        
    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
            
    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<UNK>']
        return self.word2idx[word]
    
    def __len__(self):
        return len(self.word2idx)
    

def build_vocab(captions_filename, min_count=3):
    counter = Counter()
    num_lines = len(open(captions_filename, encoding='utf8').read().split('\n'))
    with open(captions_filename, 'r', encoding='utf8') as f:
        for line in tqdm_notebook(f, total=num_lines, desc='Vocab'):
            #tokens = nltk.tokenize.word_tokenize(line.strip().lower())
            tokens = tokenizer(line.strip())
            counter.update(tokens)
            
            #if i % 1000 == 0:
            #    print("[%4d] of captions tokenized." % (i,))
                
    # discard rare words which their freguencies are less than min count
    words = [word for word, count in counter.most_common() if count >= min_count]
    
    # Create Vocabulary wrapper
    vocab = Vocabulary()
    
    # add special tokens
    vocab.add_word('<PAD>')
    vocab.add_word('<BOS>')    
    vocab.add_word('<EOS>')
    vocab.add_word('<UNK>')
    
    # add words to the vocabulary
    for word in words:
        vocab.add_word(word)
        
    return vocab
    

def main(args):
    vocab = build_vocab(json=args.caption_path,
                        threshold=args.threshold)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: %d" %len(vocab))
    print("Saved the vocabulary wrapper to '%s'" %vocab_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str, 
                        default='/media/razavi/DATA/datasets/coco2014/captions_train2014.json', 
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl', 
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=4, 
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)
