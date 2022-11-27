import torch
import torch.nn as nn
import tqdm
import os

from torch.optim import Adam
from . import VectorEmbedder

# A very naive and simple model for generating
# 1D vector embeddings. Just uses the pytorch 
# embedding layer to generate the embeddings
class NaiveVectorModel(VectorEmbedder):
    def __init__(self, tokenizer, hidden_size, vocab_size):
        super(NaiveVectorModel, self).__init__(tokenizer, hidden_size, vocab_size)
        self.hidden_size = hidden_size
        self.vocab_size  = vocab_size
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        
        self.set_optimizer(Adam(self.parameters(), lr=1e-5))

    def forward(self, x):
        return self.embeddings(x)
    
    def save(self, folder, name):
        path = os.sep.join([ folder, name ])
        state = {
            'state_dict':   self.state_dict(),
            'optimizer':    self.optimizer.state_dict(),
            'epochs':       self.num_epochs,
            'hidden_size':  self.hidden_size,
            'vocab_size':   self.vocab_size,
            'tokenizer':    self.tokenizer.save(),
            'maxlen':       self.maxlen
        }
        torch.save(state, path)
        
    @classmethod
    def load(cls, tokenizer_cls, path):
        state = torch.load(path)
        tokenizer = tokenizer_cls.load(state['tokenizer'])
        model = cls(tokenizer, state['hidden_size'], state['vocab_size'], state['maxlen'])
        model.optimizer.load_state_dict(state['optimizer'])
        model.num_epochs = state['epochs']
        model.load_state_dict(state['state_dict'])
        return model

# Usage:
# model = NaiveVectorModel(tokenizer, 256, tokenizer.vocab_size)