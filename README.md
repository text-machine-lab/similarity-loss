# Similarity-based reconstruction losses
based on the paper "Similarity-Based Reconstruction Loss for Meaning Representation", EMNLP-2018

## Description
PyTorch implementation of the autoencoder that uses word similarities for the reconstruction of the input sentences.
Parameters and file locations used for training the model are stored in a separate file `parameters.py`.
The input data is stored under the `data` folder, the trained model is saved to the `models` folder.

## Installation
```sh
git clone https://github.com/text-machine-lab/similarity-loss.git
```
```sh
pip install -r requirements.txt
```

## Requirements and dependencies
Requires PyTorch v0.4 and higher.
For installation from source please refer to:
https://github.com/pytorch/pytorch

Alternatively, install pytorch using package managers:
https://pytorch.org/get-started/locally/

For visualization purposes (optional), the Visdom library is used (https://github.com/facebookresearch/visdom).
For usage questions, please refer to the documentation . In the current implementation the visualization of the training curve is done if a port number is passed to a training function (given that the server is running).


## Word embeddings
Please note that the approach relies on using pre-trained word embeddings for computing pairwise word similarities.
As reported in the paper, the original autoencoder model was trained using fastText word embeddings available at availble at [https://fasttext.cc/docs/en/english-vectors.html].

For training an autoencoder, please specify the corresponding word vectors file location in the `parameters.py` (under `embeddings_loc` key).

## Use cases:
### Autoencoder training

For training an autoencoder, please put the training and the test .txt files under the `data/` directory. Current implementation assumes that data files contain input sentences (i.e. training examples) separated by newlines.

Run the autoencoder training as:

```sh
python main.py -loss <loss>
```

Possible values for the <loss> argument:
* `ce`: Regular cross entropy loss
* `weighted-ce`: Weighted cross entropy loss
* `weighted-sim`: Weighted similarity loss
* `soft-label`: Soft label loss

Parameters used for autoencoder training are specified in `parameters.py` and can be modified. They include files locations, the batch size, the number of epochs, the maximum sentence length, the embedding dimension, and the hidden size of the used LSTM.

### External usage
Alternatively, the losses are implemented as PyTorch Module classes and can be incorporated directly into other models.

**Notes**
- Weighted similarity loss is negative (please refer to the paper for mathematical details)
- Soft label loss considers N nearest word neighbors, where N is a model parameter. The nomalization of the encoded true-label token can be switched off by passing the `normalization=False` to the loss constructor. if `N=1` than the loss is identical to regular cross entropy. If `N` is equal to the vocabulary size, the loss becomes the _Weighted cross entropy loss_
- If the vocabulary size is substantially large, consider pruning the vocabulary (such that the word-similarity matrix can fit in the memory). The `Vocabulary` class used in the autoencoder provides the `prune()` method.
