import torch
import torch.nn as nn
import os
import tqdm

from torch.optim import Adam
from . import VectorEmbedder


class DoubleVectorModel(VectorEmbedder):
    def __init__(self, tokenizer, hidden_size, vocab_size, maxlen):
        super(DoubleVectorModel, self).__init__(tokenizer, hidden_size, vocab_size, maxlen)
        self.hidden_size = hidden_size
        self.maxlen = maxlen
        self.vocab_size = vocab_size

        self.embeddings = nn.Parameter(torch.randn(vocab_size, hidden_size))
        self.linear = nn.ModuleList([ nn.Linear(hidden_size * maxlen, hidden_size, bias=False) for _ in range(maxlen) ])
        self.set_optimizer(Adam(self.parameters(), lr=1e-3))

    def forward(self, x):
        y = self.embeddings[x]
        z = torch.reshape(y, (*y.shape[:-2], 1, self.maxlen * self.hidden_size))
        a = torch.cat([ layer(z) for layer in self.linear ], -2)
        b = torch.reshape(a, (*y.shape[:-2], self.maxlen, self.hidden_size))
        
        return b

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
# model = DoubleVectorModel(tokenizer, 256, tokenizer.vocab_size, 50)