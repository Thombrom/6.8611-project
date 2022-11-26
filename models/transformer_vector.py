from . import VectorEmbedder
from torch import nn
from torch.optim import Adam

import os
import tqdm
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
        y = self.self_attention(x)
        y = self.self_attention_aggr(y)
        y = self.layer_norm1(y + x)
        z = self.ffnn(y)
        z = self.layer_norm2(z + y)
        return z
    
# Essentially follows: https://jalammar.github.io/illustrated-transformer/
class SelfAttentionVectorModel(VectorEmbedder):
    def __init__(self, tokenizer, num_layers, hidden_size, vocab_size):
        super(SelfAttentionVectorModel, self).__init__(tokenizer, hidden_size, vocab_size)
        self.hidden_size = hidden_size
        self.vocab_size  = vocab_size
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.num_layers = num_layers
        
        # Setup optimizer
        self.set_optimizer(Adam(self.parameters(), lr=1e-5))
        
        # Setup transformer layers
        self.transformer_layer = [ TransformerVectorLayer(hidden_size, 64) for _ in range(num_layers) ]
        
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
    
    def do_train(self, dataloader, epochs, savepath, epoch_func=lambda model: None):
        loss_func = nn.NLLLoss(reduction="sum", ignore_index=self.tokenizer.PAD_TOKEN)
        while self.num_epochs <= epochs:
            
            total_loss = 0
            
            for tokens, mask_idx, replaced_token in tqdm.tqdm(dataloader, position=0, leave=True):
                tokens = tokens.to(self.device)
                mask_idx =  mask_idx.to(self.device)
                replaced_token = replaced_token.to(self.device)

                output = self.forward(tokens)
                predictions = self.generator.predict(output)
                masked_predictions = predictions[torch.arange(len(mask_idx)).unsqueeze(-1), mask_idx.unsqueeze(-1)].squeeze()
                
                loss = loss_func(masked_predictions, replaced_token)
                loss.backward()          
                self.optimizer.step()
                self.optimizer.zero_grad()
                total_loss += loss
                
            print(f"Epoch {self.num_epochs}: Total loss {total_loss}")
            epoch_func(self)
            self.num_epochs += 1
            
            if savepath:
                self.save(savepath, f"{type(self).__name__}_{self.num_epochs}.tar")