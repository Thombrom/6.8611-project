from . import VectorEmbedder, SelfAttentionVectorLayer
from torch import nn
from torch.optim import Adam

import os
import tqdm
import torch

class TransformerVectorLayer(nn.Module):
    def __init__(self, hidden_size, kvq_size = 64):
        super(TransformerVectorLayer, self).__init__()
        self.hidden_size = hidden_size
        self.kvq_size = kvq_size
        
        # Set up the parts of the transformer
        self.self_attention      = SelfAttentionVectorLayer(hidden_size, kvq_size)
        self.self_attention_aggr = nn.Linear(1 * kvq_size, hidden_size)
        self.layer_norm1         = nn.LayerNorm(hidden_size)
        self.ffnn                = nn.Linear(hidden_size, hidden_size)
        self.layer_norm2         = nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        y = self.self_attention(x)
        y = self.self_attention_aggr(y)
        y = self.layer_norm1(y + x)
        z = self.ffnn(y)
        z = self.layer_norm2(z + y)
        return z
    
# Essentially follows: https://jalammar.github.io/illustrated-transformer/
class TransformerVectorModel(VectorEmbedder):
    def __init__(self, tokenizer, num_layers, hidden_size, vocab_size):
        super(TransformerVectorModel, self).__init__(tokenizer, hidden_size, vocab_size)
        self.hidden_size = hidden_size
        self.vocab_size  = vocab_size
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.num_layers = num_layers
        
        # Setup optimizer
        self.set_optimizer(Adam(self.parameters(), lr=1e-3))
        
        # Setup transformer layers
        self.transformer_layer = nn.ModuleList([ TransformerVectorLayer(hidden_size, 64) for _ in range(num_layers) ])
        
    def forward(self, x):
        x =  self.embeddings(x)
        for idx in range(self.num_layers):
            x = self.transformer_layer[idx](x)
        return x
    
    def save(self, folder, name):
        path = os.sep.join([ folder, name ])
        state = {
            'state_dict':   self.state_dict(),
            'optimizer':    self.optimizer.state_dict(),
            'epochs':       self.num_epochs,
            'hidden_size':  self.hidden_size,
            'vocab_size':   self.vocab_size,
            'tokenizer':    self.tokenizer.save(),
            'num_layers':   self.num_layers,
        }
        torch.save(state, path)
        
    @classmethod
    def load(cls, tokenizer_cls, path):
        state = torch.load(path)
        tokenizer = tokenizer_cls.load(state['tokenizer'])
        model = cls(tokenizer, state['num_layers'], state['hidden_size'], state['vocab_size'])
        model.optimizer.load_state_dict(state['optimizer'])
        model.num_epochs = state['epochs']
        model.load_state_dict(state['state_dict'])
        return model


# Usage:
# model = TransformerVectorModel(tokenizer, 2, 256, tokenizer.vocab_size)