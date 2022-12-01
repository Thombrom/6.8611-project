from . import MatrixEmbedder, SelfAttentionMatrixLayer
from torch import nn
from torch.optim import Adam

import torch

import os
import torch
class MultiAttentionHeadsTransformerMatrixLayer(nn.Module):
    def __init__(self, hidden_shape, maxlen=None, num_attention_heads=8):
        super(MultiAttentionHeadsTransformerMatrixLayer, self).__init__()
        self.hidden_shape = hidden_shape
        self.maxlen = maxlen
        
        # Set up the parts of the transformer
        self.num_attention_heads = num_attention_heads
        self.self_attention_head_layers = nn.ModuleList([ SelfAttentionMatrixLayer(hidden_shape, maxlen) for _ in range(self.num_attention_heads)])
        self.self_attention_aggr  = nn.Conv2d(maxlen, 4 * maxlen, (3, 3), padding=1, padding_mode='zeros', groups=maxlen)
        self.linear_layer = nn.Linear(self.num_attention_heads, 1))
        self.layer_norm1          = nn.LayerNorm(self.hidden_shape)
        self.ffnn                 = nn.Conv2d(maxlen, maxlen, (3, 3), padding=1, padding_mode='zeros')
        self.layer_norm2          = nn.LayerNorm(self.hidden_shape)

    def forward(self, x):
        attention_heads = self.self_attention_head_layers[0](x)
        for i in range(1, self.num_attention_heads):
          z_i = self.self_attention_head_layers[i](x)
          attention_heads = torch.cat([attention_heads, z_i])

        # Combine multiple heads(z_i's) to one z
        y = self.linear_layer(attention_heads.permute(1, 2, 3, 0)).permute(3, 0, 1, 2)
        y = self.self_attention_aggr(y)
        
        # Transform the shape back up to the original input
        batch_size = x.shape[:-3]
        y = torch.reshape(y, (*batch_size, self.maxlen, *self.hidden_shape))
        
        # Sum x and y for layer normalization
        y = self.layer_norm1(torch.add(y, x))
        z = self.ffnn(y)
        z = self.layer_norm2(torch.add(y, z))
        return z

# Essentially follows: https://jalammar.github.io/illustrated-transformer/
# but abstracted to work for matrices
class MultiHeadsTransformerMatrixModel(MatrixEmbedder):
    def __init__(self, tokenizer, num_layers, hidden_shape, vocab_size, maxlen):
        super(MultiHeadsTransformerMatrixModel, self).__init__(tokenizer, hidden_shape, vocab_size, maxlen)
        self.hidden_shape = hidden_shape
        self.vocab_size  = vocab_size
        self.embeddings = nn.Parameter(torch.randn(vocab_size, *hidden_shape))
        self.num_layers = num_layers
        self.maxlen = maxlen
        
        # Setup optimizer
        self.set_optimizer(Adam(self.parameters(), lr=1e-3))
        
        # Setup transformer layers
        self.transformer_layer = nn.ModuleList([ MultiAttentionHeadsTransformerMatrixLayer(hidden_shape, maxlen) for _ in range(num_layers) ])
        
    def forward(self, x):
        x =  self.embeddings[x]        
        for idx in range(self.num_layers):
            x = self.transformer_layer[idx](x)
        return x
    
    def save(self, folder, name):
        path = os.sep.join([ folder, name ])
        state = {
            'state_dict':   self.state_dict(),
            'optimizer':    self.optimizer.state_dict(),
            'epochs':       self.num_epochs,
            'hidden_shape': self.hidden_shape,
            'vocab_size':   self.vocab_size,
            'tokenizer':    self.tokenizer.save(),
            'num_layers':   self.num_layers,
            'max_len':      self.maxlen
        }
        torch.save(state, path)
        
    @classmethod
    def load(cls, tokenizer_cls, path):
        state = torch.load(path)
        tokenizer = tokenizer_cls.load(state['tokenizer'])
        model = cls(tokenizer, state['num_layers'], state['hidden_shape'], state['vocab_size'], state['maxlen'])
        model.optimizer.load_state_dict(state['optimizer'])
        model.num_epochs = state['epochs']
        model.load_state_dict(state['state_dict'])
        return model
