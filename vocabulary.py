import itertools
import pickle

from collections import Counter

class Vocabulary(object):
    """
    Create a vocabulary of a training dataset.
    Args:
        text: Input text.
        If str is given, tokenization is done by RegexpTokenizer('\w+').

    Attributes:
        tokens: A set of all tokens present in the dataset
        w2idx (dict): A dictionary matching tokens to their ids
        idx2w (dict): A reversed dictionary matching ids to tokens

    """
    def __init__(self, text, tokenizer=None, bigram=False):
        if isinstance(text, str):
            if not tokenizer:
                from nltk.tokenize import RegexpTokenizer
                tokenizer = RegexpTokenizer('\w+')

            tokens = tokenizer.tokenize(text.lower())
            if bigram:
                from nltk import bigrams
                tokens = list(bigrams(tokens))
            self.counts = Counter(tokens)
            tokens = list(set(tokens))

        else:
            raise TypeError('Invalid type of input data. Acceptable type is string.')

        self.tokens = tokens
        self.w2idx = {word: idx for idx, word in enumerate(self.tokens)}
        self.idx2w = {idx: word for word, idx in self.w2idx.items()}

        assert len(self.w2idx) == len(self.idx2w)

    def add_token(self, token):
        """

        Args:
            token: a token to be added

        Returns: updated vocabulary

        """
        if token in self.w2idx.keys():
            print('Token "{}" is already present in the vocabulary'.format(token))
            return self
        cur_len = self.get_length()
        self.tokens.append(token)
        self.w2idx[token] = cur_len
        self.idx2w[cur_len] = token

        assert len(self.w2idx) == len(self.idx2w)
        return self

    def get_length(self):
        """
        Get the current vocabulary size.
        Returns (int): Vocabulary size

        """
        return len(self.tokens)

    def prune(self, threshold=10):
        self.tokens = list(token for token in self.tokens if self.counts[token] >= threshold)
        self.w2idx = {word: idx for idx, word in enumerate(self.tokens)}
        self.idx2w = {idx: word for word, idx in self.w2idx.items()}
        assert len(self.w2idx) == len(self.idx2w)

    def save(self, loc):
        with open(loc, 'wb') as f:
            pickle.dump(self, f)
        print('Vocabulary saved to {}'.format(loc))
