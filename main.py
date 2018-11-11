import argparse
import numpy as np
import os.path
import pickle
import random

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from autoencoder import Autoencoder
from dataset import AutoencoderDataset
from loss import SoftLabelLoss, WeightedCrossEntropyLoss, WeightedSimilarityLoss
from train import train
from util import cuda, match_embeddings
from vocabulary import Vocabulary


INIT_TOKEN = '<start>'
UNK_TOKEN = '<unk>'
END_TOKEN = '<end>'
PAD_TOKEN = '<pad>'


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-loss", help="loss function used. "
                                     "0 is cross-entropy;"
                                     "1 is weighted cross-entropy;"
                                     "2 is weighted similarity;"
                                     "3 is soft-label")
    args = vars(parser.parse_args())

    # PARAMETERS #
    max_len = 15
    hidden_size = 256
    embedding_dim = 300
    batch_size = 300
    n_epochs = 1000

    # DATA FILES #
    train_loc = 'yelp/merged/train'
    dev_loc = 'yelp/merged/dev'
    fasttext_loc = '/data1/word_vectors/fastText/crawl-300d-2M.vec'
    w2vec_loc = 'vocabulary/word_to_vec.pkl'
    model_loc = 'model.pt'
    stopwordsfile = 'stop_words.txt'

    # VOCABULARY #
    special_tokens = [INIT_TOKEN, UNK_TOKEN, END_TOKEN, PAD_TOKEN]
    with open(train_loc) as f:
        raw_text = f.read()
    voc = Vocabulary(raw_text)
    for token in special_tokens:
        voc.add_token(token)
    w2idx = voc.w2idx
    idx2w = voc.idx2w
    voc_size = voc.get_length()
    pad_idx = w2idx[PAD_TOKEN]
    init_idx = w2idx[INIT_TOKEN]

    # STOP WORDS #
    with open(stopwordsfile) as f:
        stop_words = f.read().split()
    stop_words.extend(special_tokens)
    stop_idx = [w2idx[w] for w in stop_words if w in w2idx.keys()]

    # PRE-TRAINED EMBEDDINGS #
    if os.path.exists(w2vec_loc):
        with open(w2vec_loc, 'rb') as f:
            w2vec = pickle.load(f)
    else:
        w2vec = {}
        print("Loading word vectors...")
        with open(fasttext_loc) as f:
            f.__next__()
            for line in tqdm(f):
                items = line.strip().split(' ')
                token = items[0]
                vector = np.array(items[1:]).astype(float)
                w2vec[token] = vector
        with open(w2vec_loc, 'wb') as f:
            pickle.dump(w2vec, f)
    dim = len(random.choice(list(w2vec.values())))
    embeddings = cuda(torch.FloatTensor(match_embeddings(idx2w, w2vec, dim)))

    # DATASET #
    dataset_train = AutoencoderDataset(train_loc, voc, max_len)
    dataset_dev = AutoencoderDataset(dev_loc, voc, max_len)

    dataloader_train = DataLoader(dataset_train, batch_size, shuffle=True)
    dataloader_dev = DataLoader(dataset_dev, batch_size, shuffle=True)
    dataloaders = {'train': dataloader_train, 'dev': dataloader_dev}

    # MODEL #
    model = cuda(Autoencoder(hidden_size, voc_size, pad_idx, init_idx, max_len, embeddings=embeddings))
    optimizer = optim.Adam([parameter for parameter in list(model.parameters()) if parameter.requires_grad], lr=0.00005)

    if args['loss'] == '0':
        criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    elif args['loss'] == '1':
        criterion = WeightedCrossEntropyLoss(embeddings, ignore_idx=pad_idx)
    elif args['loss'] == '2':
        criterion = WeightedSimilarityLoss(embeddings, ignore_idx=pad_idx)
    elif args['loss'] == '3':
        criterion = SoftLabelLoss(stop_idx, embeddings, N=5, ignore_idx=pad_idx)

    # TRAIN #
    train(model, model_loc, criterion, optimizer, dataloaders, n_epochs, idx2w, pad_idx, port=8098)





