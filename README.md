# Similarity-based reconstruction losses
based on the paper "Kovaleva, O., Rumshisky, A. and Romanov, A., 2018. Similarity-Based Reconstruction Loss for Meaning Representation. EMNLP 2018" [PDF](http://www.aclweb.org/anthology/D18-1525) [BibTex](https://bit.ly/2zufchE)


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
Requires PyTorch v0.4.0.
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

For training an autoencoder, please put the training and the test .txt files under the `data/` directory. If the name of the training file is `train.txt` and the name of the test file is `test.txt` then the `train_loc` and the `test_loc` key of the `params` dict in `parameters.py` should be set to `data/train.txt` and `data/test.txt` respectively. Current implementation assumes that data files contain input sentences (i.e. training examples) separated by newlines. 

Run the autoencoder training as:

```sh
python main.py -loss <loss>
```

Possible values for the <loss> argument:
* `ce`: Regular cross entropy loss
* `weighted-ce`: Weighted cross entropy loss
* `weighted-sim`: Weighted similarity loss
* `soft-label`: Soft label loss

Parameters used for autoencoder training are specified in `parameters.py` and can be modified. They include the used files locations, the batch size, the number of epochs, the maximum sentence length, the embedding dimension, and the hidden size of the encoding (and the decoding) LSTM.

### External usage
Alternatively, the losses are implemented as PyTorch Module classes and can be incorporated directly into other models.

You can replace the conventional `nn.CrossEntropy()` loss with any of the three loss classes implemented in `loss.py`, namely 
`WeightedSimilarityLoss()`, `WeightedCrossEntropyLoss()` and `SoftLabelLoss()`. 
1. Instantiate any of the classes in your training script, e.g.:
```python
criterion = SoftLabelLoss(stop_idx, embeddings, N=N, ignore_idx=pad_idx)
```
2. Compute the loss by running the forward propagation, e.g.:
```python
outputs = model(inputs)
loss = criterion(outputs, targets)
```
3. Compute the gradients as:
```python
loss.backward()
```
Note that every loss class takes a tensor of word embeddings as input (for computing of the word similarities across the vocabulary). Also note that `SoftLabelLoss` takes two additional parameters: `N` for limiting the consideration of word neighbors to only top N closest ones and `stop_idcs` for encoding stop words using a traditional one hot encoding scheme. For more details about every loss function please refer to the paper, to the implementation and to the notes below.

## Notes
* The equations defining each of losses are as follows:
  * Weighted similarity loss:  
    <img src="https://latex.codecogs.com/svg.latex?\mathcal{L}&space;=&space;-\sum_{i=1}^V&space;\text{sim}(y_t,&space;y_i)p_i" title="\mathcal{L} = -\sum_{i=1}^V \text{sim}(y_t, y_i)p_i" />
  * Weighted cross entropy loss:  
    <img src="https://latex.codecogs.com/svg.latex?\mathcal{L}&space;=&space;-\sum_{i=1}^V&space;\text{sim}(y_t,&space;y_i)&space;\log&space;p_i" title="\mathcal{L} = -\sum_{i=1}^V \text{sim}(y_t, y_i) \log p_i" />
  * Soft label loss:  
    <img src="https://latex.codecogs.com/svg.latex?\mathcal{L}&space;=&space;-\sum_{i=1}^V&space;y_i^*log{p_i}" title="\mathcal{L} = -\sum_{i=1}^V y_i^*log{p_i}" />  
    <img src="https://latex.codecogs.com/svg.latex?y_i^*&space;=&space;\begin{cases}\frac{\text{sim}(y_t,&space;y_i)}{\sum_{j=1}^N&space;\text{sim}(y_t,&space;y_j)},&space;&&space;y_i&space;\in&space;\text{top&space;N}&space;\\&space;0,&space;&&space;y_i&space;\not&space;\in&space;\text{top&space;N}&space;\end{cases}" title="y_i^* = \begin{cases}\frac{\text{sim}(y_t, y_i)}{\sum_{j=1}^N \text{sim}(y_t, y_j)}, & y_i \in \text{top N} \\ 0, & y_i \not \in \text{top N} \end{cases}" />
* Weighted similarity loss is negative (please refer to the paper for details)
* Soft label loss considers N nearest word neighbors, where N is a model parameter. The nomalization of the encoded true-label token can be switched off by passing the `normalization=False` to the loss constructor. if `N=1` than the loss is identical to regular cross entropy. If `N` is equal to the vocabulary size, the loss becomes the `WeightedCrossEntropyLoss`
* If the vocabulary size is substantially large, consider pruning the vocabulary (such that the word-similarity matrix can fit in the memory). The `Vocabulary` class used in the autoencoder provides the `prune()` method.




