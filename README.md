#

## Description


## Autoencoder 
Run the autoencoder training as 

```sh
python main.py -loss <loss>
```

Possible values for the loss argument:
0: Regular cross entropy loss
1: Weighted cross entropy loss
2: Weighted similarity loss
3: Soft label loss

## External usage

Alternatively, the losses are implemented as PyTorch Module and can be incorporated directly into other models.

Note that word similarities are computed using pre-trained fastText embeddings availble at ...
Other word vectors can be used instead (the location of the file should be spcified).

