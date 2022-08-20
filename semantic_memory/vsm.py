"""Build Vector Space Models"""
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from typing import List, Tuple, Union, Callable
from .vsm_utils import cosine, jaccard


class VectorSpaceModel(object):
    """
    Class that initializes a n x m dimensional vector space
    with named vectors (words/concepts/senses/etc.)
    """

    def __init__(self, name: str, dimensions: int = None) -> None:
        self.name: str = name
        self.embeddings: dict = None
        self.vocab: list = []
        self.dimensions: int = dimensions
        self.vocab2idx = defaultdict(lambda: len(self.vocab2idx))
        self.vocab_size: int = None

    def __repr__(self) -> str:
        return f"<{self.name} VectorSpaceModel: {self.vocab_size} x {self.dimensions}>"

    def __call__(self, word: Union[List, str]) -> torch.Tensor:
        words = [word] if isinstance(word, str) else word
        key = [self.vocab2idx[w] for w in words]
        return self.embeddings[key]

    def load_vectors_from_tensor(self, weights, vocab) -> None:
        self.embeddings = weights
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.dimensions = self.embeddings.shape[1]

        for item in vocab:
            _ = self.vocab2idx[item]
        self.vocab2idx.default_factory = None
        self.idx2vocab = {v: k for k, v in self.vocab2idx.items()}

        self.shape = self.embeddings.shape

    def load_vectors(
        self, file, data_type="float32", quotes=False, ignore_first=False
    ) -> None:
        self.embeddings = {}
        with open(file) as f:
            if ignore_first:
                rows, cols = f.readline().strip().split(" ")
                self.dimensions = int(cols)
                self.vocab_size = int(rows)
            for i, line in enumerate(tqdm(f)):
                values = line.split()
                if self.dimensions is None:
                    dimensions = len(values) - 1
                else:
                    dimensions = self.dimensions
                item = "".join(values[:-dimensions])
                if quotes:
                    item = item.replace('"', "").replace("'", "")
                self.vocab2idx[item]
                vector = np.asarray(values[-dimensions:], dtype=data_type)
                self.embeddings[item] = vector
        if self.dimensions is None:
            self.dimensions = dimensions

        self.embeddings = torch.tensor(np.stack(list(self.embeddings.values())))
        self.vocab = list(self.vocab2idx.keys())
        if self.vocab_size is None:
            self.vocab_size = i + 1

        self.vocab2idx.default_factory = None
        self.shape = self.embeddings.shape
        self.idx2vocab = {v: k for k, v in self.vocab2idx.items()}

    def neighbor(
        self,
        word: Union[list, str, torch.Tensor],
        k: int,
        space: list = None,
        names_only=False,
        ignore_first: bool = True,
        nearest=True,
        sim_function: Callable = cosine,
    ) -> List:

        if isinstance(word, list) or isinstance(word, str):
            words = [word] if isinstance(word, str) else word
            idx = [self.vocab2idx[w] for w in words]
            query = self.embeddings[idx]
        elif isinstance(word, torch.Tensor):
            query = word
        else:
            raise TypeError("Only accepts list, string, or nxd tensors!")
        if space is not None:
            space_idx = [self.vocab2idx[w] for w in space]
            # idx2vocab = {k:self.idx2vocab[k] for k in [self.vocab2idx[x] for x in space]}
            idx2vocab = {k: v for k, v in enumerate(space)}
        else:
            space_idx = range(self.vocab_size)
            idx2vocab = self.idx2vocab
        similarities = sim_function(query, self.embeddings[space_idx])
        # by default always ignore first element as it will be the same.
        if nearest:
            if ignore_first:
                topk = similarities.topk(k + 1)
                values = topk.values[:, None][:, :, 1:].squeeze().tolist()
                indices = topk.indices[:, None][:, :, 1:].squeeze()
            else:
                topk = similarities.topk(k)
                values = topk.values.tolist()
                indices = topk.indices
        else:
            # farthest neighbors
            topk = (1.0 - similarities).topk(k)
            values = topk.values.tolist()
            indices = topk.indices

        if len(indices.shape) == 0:
            names = idx2vocab[indices.item()]
            if names_only:
                neighbors = names
            else:
                neighbors = [(names, values)]

        elif len(indices.shape) == 1:
            ## what is this?
            names = [idx2vocab[i] for i in indices.tolist()]
            if names_only:
                neighbors = names
            else:
                neighbors = list(zip(names, values))

        else:
            names = [[idx2vocab[i] for i in bunch] for bunch in indices.tolist()]

            if names_only:
                neighbors = names
            else:
                neighbors = [list(zip(name, sim)) for name, sim in zip(names, values)]
        return neighbors

    def pairwise(self, words: list, sim_function: Callable = cosine) -> torch.Tensor:
        assert len(words) > 1

        idx = [self.vocab2idx[w] for w in words]
        query = self.embeddings[idx]
        sim_matrix = sim_function(query, query)
        return sim_matrix

    def from_tensor(self, vectors: torch.Tensor, vocab: list) -> None:
        raise NotImplementedError
