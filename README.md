# Similarity-based reconstruction losses
based on the paper "Similarity-Based Reconstruction Loss for Meaning Representation", EMNLP-2018

## Description
PyTorch implementation of the autoencoder that uses word similarities for the reconstruction of the input sentences.
Parameters used for training the model are stored in a separate file `parameters.py`.
The input data is stored under the `data` folder, the trained model is saved to the `models` folder.

## Requirements
Requires PyTorch v0.4 and higher.


## Autoencoder 
Run the autoencoder training as 

```sh
python main.py -loss <loss>
```

Possible values for the <loss> argument:
*0: Regular cross entropy loss
*1: Weighted cross entropy loss
*2: Weighted similarity loss
*3: Soft label loss

## External usage

Alternatively, the losses are implemented as PyTorch Module and can be incorporated directly into other models.

Note that word similarities are computed using pre-trained fastText embeddings availble at ...
Other word vectors can be used instead (the location of the file should be spcified).

