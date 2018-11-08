import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from util import cuda


class WeightedSimilarityLoss(nn.Module):
    """
    The per-example loss takes predicted probabilities for every word in a vocabulary and penalizes the model
    for predicting high probabilities for words that are semantically distant from the target word.
    Note that instead of log-probabilities the loss operates on probabilities themselves.

    Args:
        embeddings (torch.FloatTensor): pre-trained word embeddings
        ignore_idx (int): vocabulary id corresponding to the padding token
    """
    def __init__(self, embeddings, ignore_idx):
        super().__init__()
        voc_size = embeddings.shape[0]

        # Compute similarities
        print("Computing word similarities...")
        similarities = []
        for i in tqdm(range(voc_size)):
            similarities.append(F.cosine_similarity(embeddings[i].expand_as(embeddings), embeddings))
        similarities = cuda(torch.stack(similarities))

        # Ignore padding index penalties
        similarities[ignore_idx] = torch.zeros(voc_size)
        self.similarities = similarities

    def forward(self, output, targets, eps=1e-5):
        # Calculate the loss based on the computed similarities
        softmax = nn.Softmax(dim=1)
        probs = softmax(output) + eps
        loss = torch.mean(torch.sum(-probs * self.similarities[targets], 1))
        return loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    The per-example loss takes predicted probabilities for every word in a vocabulary and penalizes the model
    for predicting high probabilities for words that are semantically distant from the target word.
        Args:
        embeddings (torch.FloatTensor): pre-trained word embeddings
        ignore_idx (int): vocabulary id corresponding to the padding token
    """
    def __init__(self, embeddings, ignore_idx):
        super().__init__()
        voc_size = embeddings.shape[0]

        # Compute similarities
        print("Computing word similarities...")
        similarities = []
        for i in tqdm(range(voc_size)):
            similarities.append(F.cosine_similarity(embeddings[i].expand_as(embeddings), embeddings))
        similarities = cuda(torch.stack(similarities))

        # Ignore padding index penalties
        similarities[ignore_idx] = torch.zeros(voc_size)
        self.similarities = similarities

    def forward(self, output, targets, eps=1e-5):
        softmax = nn.Softmax(dim=1)
        probs = softmax(output) + eps
        loss = torch.mean(torch.sum(-torch.log(probs)*self.similarities[targets], 1))
        return loss


class SoftLabelLoss(nn.Module):
    """
    The loss "encodes" ground-truth tokens as their similarities across the vocabulary,
    but the consideration is limited to only the top N closest words in the vocabulary.
    Stop-words are encoded using the traditional one-hot-encoding scheme.

    Args:
        stop_idcs (list): list of stop word ids in the vocabulary
        embeddings (torch.FloatTensor): pre-trained word embeddings
        ignore_idx (int): vocabulary id corresponding to the padding token
        N (int): number of vocabulary neighbors considered
        normalization (boolean): if True, resulting labels over vocabulary add up to 1
    """
    def __init__(self, stop_idcs, embeddings, ignore_idx, N=5, normalization=True):
        super().__init__()
        voc_size = embeddings.shape[0]
        all_targets = []
        print("Computing word similarities...")
        for word_idx in tqdm(range(voc_size)):
            target = torch.zeros(voc_size)
            if word_idx != ignore_idx:
                if word_idx not in stop_idcs:
                    embedding = embeddings[word_idx]
                    # Compute similarities
                    similarities = F.cosine_similarity(embedding.expand_as(embeddings), embeddings)

                    # Get top N word neighbors with their similarities
                    similarities, indices = torch.sort(similarities, descending=True)
                    indices = indices[:N]
                    similarities = similarities[:N]

                    # Normalize computed similarities
                    if normalization:
                        normalization_factor = torch.sum(similarities)
                    else:
                        normalization_factor = 1
                    weights = similarities / normalization_factor
                    for i, idx in enumerate(indices):
                        target[idx] = weights[i]
                else:
                    # Ignore padding index penalties
                    target[word_idx] = 1
            all_targets.append(target)
        soft_targets = cuda(torch.stack(all_targets))
        self.soft_targets = soft_targets

    def forward(self, output, targets, eps=1e-5):
        # Extract "soft-labels" based on true targets
        soft_targets = self.soft_targets[targets]

        # Compute predictions and the loss
        softmax = nn.Softmax(dim=1)
        pred = softmax(output) + eps
        loss = torch.mean(torch.sum(- soft_targets * torch.log(pred), 1))
        return loss

