import torch
import torch.nn as nn
import torch.nn.functional as F

# A very naive and simple model for generating
# 1D vector embeddings. Just uses the pytorch 
# embedding layer to generate the embeddings
class NaiveModel(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        self.hidden_size = hidden_size
        self.vocab_size  = vocab_size
        self.embeddings = nn.Embedding(vocab_size, hidden_size)

    def forward(self, x):
        return self.embeddings(x)