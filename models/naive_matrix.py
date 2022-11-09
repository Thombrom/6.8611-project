import torch
import torch.nn as nn
from . import MatrixEmbedder

class NaiveMatrixModel(MatrixEmbedder):
    def __init__(self, tokenizer, shape, vocab_size):
        super(NaiveMatrixModel, self).__init__(tokenizer, shape, vocab_size)
        self.shape = shape
        self.vocab_size = vocab_size
        self.embeddings = nn.Parameter(torch.randn(vocab_size, *shape))
        
    def forward(self, x):
        return self.embeddings[x]
    