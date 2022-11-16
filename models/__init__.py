import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.optim import Adam

class Generator():
    def __init__(self):
        pass

class VectorGenerator(Generator):
    """A generator for vectors"""
    def __init__(self, hidden_size, vocab_size):
        super(VectorGenerator, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size  = vocab_size
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def predict(self, x):
        """Predict a word in the vocab, this is as linear projection"""
        return F.log_softmax(self.proj(x), dim=-1)

    def vectorize(self, x):
        """Turning the input into a vector - this is simply the identity"""
        return x

class MatrixGenerator(Generator):
    """A generator for matrices"""
    def __init__(self, shape, vocab_size):
        super(MatrixGenerator, self).__init__()
        self.shape = shape
        self.vocab_size = vocab_size
        self.proj = nn.Conv2d(1, vocab_size, shape, bias=False, padding=0)
        
    def predict(self, x):
        top_shape = x.shape[:-2]
        y = x.reshape(np.prod(x.shape[:-2]), *x.shape[-2:])        
        z = torch.unsqueeze(y, 1)
        proj = self.proj(z).squeeze(-1).squeeze(-1)
        return F.log_softmax(proj, dim=-1).reshape(*top_shape, self.vocab_size)
        
    def vectorize(self, x):
        new_shape = x.shape[-1] * x.shape[-2]
        new_shape = x.shape[:-2] + (new_shape,)
        return x.reshape(new_shape)

class Embedder(nn.Module):
    def __init__(self, generator, tokenizer):
        super(Embedder, self).__init__()
        self.generator  = generator
        self.tokenizer  = tokenizer
        self.num_epochs = 0
        self.optimizer  = None
        
    def set_num_epochs(self, num):
        self.num_epochs = num
        
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
        
class VectorEmbedder(Embedder):
    def __init__(self, tokenizer, hidden_size, vocab_size):
        super(VectorEmbedder, self).__init__(VectorGenerator(hidden_size, vocab_size), tokenizer)


class MatrixEmbedder(Embedder):
    def __init__(self, tokenizer, shape, vocab_size):
        super(MatrixEmbedder, self).__init__(MatrixGenerator(shape, vocab_size), tokenizer)    