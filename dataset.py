from torch.utils.data import Dataset

from util import encode_sentence


class AutoencoderDataset(Dataset):
    """
    Args:
        filename (str): .txt file containing training examples on separate lines
        voc (obj, optional): Vocabulary object that has w2idx and idx2w attributes
        max_len (int, optional): Maximum sentence length (a sentence is cropped if its size exceeds max_len).
        If no value provided, the maximum sentence length in the dataset is used.
    Attributes:
        w2idx (dict): A dictionary matching tokens to their ids
        idx2w (dict): A reversed dictionary matching ids to tokens
        data (list): List of input lines
    """

    def __init__(self, filename, voc, max_len=None):
        super(AutoencoderDataset, self).__init__()

        # .txt file with separate sentences on separate lines
        with open(filename) as f:
            self.data = f.readlines()

        # For padding to the same length
        if max_len is not None:
            self.max_len = max_len
        else:
            self.max_len = self.get_max_example_len()

        self.w2idx = voc.w2idx
        self.idx2w = voc.idx2w

    def get_max_example_len(self):
        from nltk.tokenize import RegexpTokenizer
        tokenizer = RegexpTokenizer('\w+')
        max_len = max(map(len, map(tokenizer.tokenize, self.data)))
        return max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data[idx]
        enc_sentence = encode_sentence(sentence, self.w2idx, self.max_len)
        return enc_sentence
