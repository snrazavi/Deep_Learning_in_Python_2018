import pickle
import numpy as np


class WordVector(object):
    def __init__(self, lang, filename):
        " Create an instance of word vecor class."
        self.lang = lang
        self.word_vectors = pickle.load(open(filename, 'rb'))
        self.vector_size = len(self.word_vectors['</s>'])
        self.mean_vector = None
        self.stddev = None
        
    def __len__(self):
        return len(self.word_vectors)
    
    def __getitem__(self, word):
        return self.get_word_vector(word)
        
    def get_words(self):
        return [w for w in self.word_vectors]
    
    def mean(self):
        if self.mean_vector is None:
            vectors = np.stack([v for _, v in self.word_vectors.items()], axis=0)
            self.mean_vector = np.mean(vectors, axis=0)
        return self.mean_vector
    
    def std(self):
        if self.stddev is None:
            vectors = np.stack([v for _, v in self.word_vectors.items()], axis=0)
            self.stddev = np.std(vectors)
        return self.stddev
    
    def get_word_vector(self, word):
        return self.word_vectors.get(word, self.mean() + np.random.normal(scale=self.std(), size=(self.vector_size,)))
        
    def get_word_vectors(self, words):
        return [self.get_word_vector(w) for w in words]
        
    def most_similar(self, word, n=10):
        v = self.get_word_vector(word)
        V = np.stack([v for _, v in self.word_vectors.items()], axis=0)
        dist = V - v
        idxes = np.argsort(np.sum(dist * dist, axis=1))[:n]
        
        result = []
        for i, w in enumerate(self.word_vectors.keys()):
            if i in idxes:
                result += [w]
        return result