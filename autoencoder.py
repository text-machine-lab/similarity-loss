import torch
import torch.nn as nn
import torch.nn.functional as F

from util import get_sequences_lengths, variable, cuda


class Autoencoder(nn.Module):
    """
    Args:
        hidden_size (int): Hidden size of LSTM
        voc_size (int): Vocabulary size
        padding_idx (int): Index of a padding token in the vocabulary
        init_idx (int): Index of a start-of-sentence token in the vocabulary
        max_len (int): Maximum length of the sentence
        embeddings (torch.FloatTensor, optional): Tensor of pre-trained word embeddings
        embedding_dim (int, optional): Dimension of a trained embedding layer if pre-trained embeddings are not provided.
        Defaults to 300
    """
    def __init__(self, hidden_size, voc_size, padding_idx, init_idx, max_len, embeddings=None, embedding_dim=300):
        super().__init__()

        # Sizes
        if embeddings is not None:
            self.embedding_dim = embeddings.shape[1]
        else:
            self.embedding_dim = embedding_dim

        self.hidden_size = hidden_size
        self.voc_size = voc_size
        self.max_len = max_len

        # Indices
        self.init_idx = init_idx
        self.padding_idx = padding_idx

        # Layers
        if embeddings is not None:
            self.embeddings = cuda(embeddings)
            self.emb = nn.Embedding.from_pretrained(self.embeddings, freeze=False)
        else:
            self.emb = nn.Embedding(self.voc_size, self.embedding_dim)
        self.enc = nn.LSTM(self.embedding_dim, self.hidden_size, batch_first=True)
        self.dec = nn.LSTMCell(self.embedding_dim, self.hidden_size)
        self.lin = nn.Linear(self.hidden_size, self.voc_size)
        self.dropout = nn.Dropout(p=0.3)

    def encoder(self, inputs):
        # Get lengths
        lengths = get_sequences_lengths(inputs, pad_idx=self.padding_idx)

        # Sort as required for pack_padded_sequence input
        lengths, indices = torch.sort(lengths, descending=True)
        inputs = inputs[indices]

        # Pack
        inputs = torch.nn.utils.rnn.pack_padded_sequence(self.emb(inputs), lengths.data.tolist(), batch_first=True)

        # Encode
        outputs, (hidden, _) = self.enc(inputs)

        # Unsort in the original order
        _, unsort_ind = torch.sort(indices)
        hidden = hidden.squeeze(0)[unsort_ind]
        return hidden

    def decoder(self, last_hidden, targets):
        batch_size = last_hidden.shape[0]
        outputs = []

        # Initialize decoder states
        cell = variable(torch.zeros(batch_size, last_hidden.shape[1]))
        hidden = last_hidden

        # Initialize generated steps with a start token
        step = variable(torch.LongTensor(batch_size, ).fill_(self.init_idx))

        # Generating loop
        for i in range(self.max_len):
            hidden, cell = self.dec(self.emb(step), (hidden, cell))
            output = self.lin(hidden)
            if targets is not None and i < len(targets):
                step = targets[:, i]
            else:
                step = torch.max(F.softmax(output, dim=-1), 1)[1]
            outputs.append(output)

        outputs = torch.stack(outputs, dim=1)
        return outputs

    def forward(self, inputs):
        if self.training:
            targets = inputs
        else:
            targets = None

        hidden = self.encoder(inputs)
        reg_hidden = self.dropout(hidden)
        outputs = self.decoder(reg_hidden, targets)
        return outputs

