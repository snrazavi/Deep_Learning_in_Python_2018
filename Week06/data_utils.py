class Vocabulary(object):
    
    def __init__(self):
        self.word2index = {}
        self.index2word = {}
        self.word2count = {}
        self.num_words = 0
        
    def add_word(self, word):
        if not word in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1
        
    def add_words(self, words):
        for word in words:
            self.add_word(word)
    
    def __len__(self):
        return self.num_words