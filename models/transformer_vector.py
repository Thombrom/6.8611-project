from . import VectorEmbedder
from torch import nn

import torch

class SelfAttentionVectorLayer(nn.Module):
    def __init__(self, hidden_size, kvq_size = 64):
        super(SelfAttentionVectorLayer, self).__init__()
        self.hidden_size = hidden_size
        self.kvq_size = kvq_size
        
        # Set up key, query and value projections
        self.key_proj   = nn.Linear(hidden_size, kvq_size, bias=False)
        self.query_proj = nn.Linear(hidden_size, kvq_size, bias=False)
        self.value_proj = nn.Linear(hidden_size, kvq_size, bias=False)
        
    def forward(self, x):
        key   = self.key_proj(x)
        value = self.value_proj(x)
        query = self.query_proj(x)
                
        attention = torch.bmm(query, torch.transpose(key, 1, 2))
        attention /= self.kvq_size**(1/2)
        attended_value = torch.bmm(attention, value)
        
        return attended_value

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
        x = self.self_attention(x)
        x = self.self_attention_aggr(x)
        x = self.layer_norm1(x)
        x = self.ffnn(x)
        x = self.layer_norm2(x)
        return x
    
# Essentially follows: https://jalammar.github.io/illustrated-transformer/
class SelfAttentionVectorModel(VectorEmbedder):
    def __init__(self, tokenizer, hidden_size, vocab_size):
        super(SelfAttentionVectorModel, self).__init__(tokenizer, hidden_size, vocab_size)
        self.hidden_size = hidden_size
        self.vocab_size  = vocab_size
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        
        self.transformer_layer = TransformerVectorLayer(hidden_size, 64)
        
    def forward(self, x):
        x =  self.embeddings(x)
        return self.transformer_layer(x)