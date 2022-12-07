import torch
import torch.nn as nn
import os
import tqdm

from torch.optim import Adam
from . import MatrixEmbedder


class DoubleVectorModel(MatrixEmbedder):
    def __init__(self, tokenizer, shape, vocab_size, maxlen):
        super(DoubleVectorModel, self).__init__(tokenizer, shape, vocab_size, maxlen)
        self.shape = shape
        self.maxlen = maxlen
        self.vocab_size = vocab_size

        self.embeddings = nn.Parameter(torch.randn(vocab_size, (shape,1)))
        self.linear = nn.Linear(maxlen, maxlen, bias=False)
        self.set_optimizer(Adam(self.parameters(), lr=1e-3))

    def forward(self, x):
        x = self.embeddings[x]
        return self.linear(x)

    def save(self, folder, name):
        path = os.sep.join([folder, name])
        state = {
            'state_dict': self.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epochs': self.num_epochs,
            'shape': self.shape,
            'vocab_size': self.vocab_size,
            'tokenizer': self.tokenizer.save(),
            'maxlen': self.maxlen
        }
        torch.save(state, path)

    @classmethod
    def load(cls, tokenizer_cls, path):
        state = torch.load(path)
        tokenizer = tokenizer_cls.load(state['tokenizer'])
        model = cls(tokenizer, state['shape'], state['vocab_size'], state['maxlen'])
        model.optimizer.load_state_dict(state['optimizer'])
        model.num_epochs = state['epochs']
        model.load_state_dict(state['state_dict'])
        return model

# Usage:
model = DoubleMatrixModel(tokenizer, 16, tokenizer.vocab_size, 50)