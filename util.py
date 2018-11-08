import logging
import numpy as np
import torch

from nltk.tokenize import RegexpTokenizer
from torch.autograd import Variable
from tqdm import tqdm


def get_sequences_lengths(sequences, pad_idx, dim=1):
    if len(sequences.size()) > 2:
        sequences = sequences.sum(dim=2)

    masks = torch.ne(sequences, pad_idx)

    lengths = masks.sum(dim=dim)

    return lengths


def cuda(obj):
    if torch.cuda.is_available():
        obj = obj.cuda()

    return obj


def variable(obj, volatile=False):
    if isinstance(obj, (list, tuple)):
        return [variable(o, volatile=volatile) for o in obj]

    if isinstance(obj, np.ndarray):
        obj = torch.from_numpy(obj)

    obj = cuda(obj)
    obj = Variable(obj, volatile=volatile)
    return obj


def match_embeddings(idx2w, w2vec, dim):
    embeddings = []
    voc_size = len(idx2w)
    print("Matching embeddings to vocabulary ids...")
    for idx in tqdm(range(voc_size)):
        word = idx2w[idx]
        if word not in w2vec:
            embeddings.append(np.random.uniform(low=-1.2, high=1.2, size=(dim, )))
        else:
            embeddings.append(w2vec[word])

    embeddings = np.stack(embeddings)
    return embeddings


def idx2text(token_ids, idx2w):
    """
    Convert a list of token ids into a text string.
    Args:
        token_ids (list): Input list of token ids
        idx2w (dict): A reversed dictionary matching indices to tokens

    Returns (str): Text string

    """
    sent = ""
    for id in token_ids:
        if idx2w[id] == '<end>':
            break
        if idx2w[id] != '<pad>':
            sent += idx2w[id] + ' '
    return sent


def save_weights(model, filename):
    if not isinstance(filename, str):
        filename = str(filename)

    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    torch.save(model.state_dict(), filename)
    logging.info('Model saved: {os.path.basename(filename)}')


def restore_weights(model, filename):

    if not isinstance(filename, str):
        filename = str(filename)
    map_location = None
    # load trained on GPU models to CPU
    if not torch.cuda.is_available():
        map_location = lambda storage, loc: storage
    state_dict = torch.load(filename, map_location=map_location)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.load_state_dict(state_dict)
    logging.info('Model restored: {os.path.basename(filename)}')
    return


def encode_sentence(sentence, w2idx, max_len, pad_token='<pad>', end_token='<end>', tokenizer=None):
    """
    Process the sequence of ids by padding it to a specified length and adding an ending token
    Args:
        sentence (list): A list of tokens
        w2idx (dict): A dictionary matching tokens to their ids
        max_len: Maximum length for padding. Longer sentences are cropped
        pad_token: Padding token
        end_token: Ending token

    Returns (numpy array): Transformed sentence

    """
    enc_sentence = text2idx(sentence, w2idx, tokenizer)
    if len(enc_sentence) > max_len - 1:
        enc_sentence = enc_sentence[:max_len-1]
    enc_sentence = enc_sentence + [w2idx[end_token]]
    enc_sentence = enc_sentence + [w2idx[pad_token]]*(max_len - len(enc_sentence))
    enc_sentence = np.array(enc_sentence)
    return enc_sentence


def text2idx(words, w2idx, tokenizer=None):
    """
    Convert a string or an iterable of tokens into token ids according to the vocabulary.
    Args:
        words (str or iterable): Input string or an iterable of tokens.
        If a string is used, the default tokenizer is RegexpTokenizer('w+')

        w2idx (dict):  A dictionary matching tokens to their ids

    Returns: a list of token ids

    """
    if isinstance(words, str):
        words = words.lower()
        if not tokenizer:
            tokenizer = RegexpTokenizer('\w+')
        words = tokenizer.tokenize(words)
    ids = []
    for word in words:
        if word in w2idx.keys():
            ids.append(w2idx[word])
        else:
            ids.append(w2idx['<unk>'])
    return ids