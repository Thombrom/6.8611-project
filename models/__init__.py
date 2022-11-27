import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import tqdm
from torch.optim import Adam

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

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
    def __init__(self, generator, tokenizer, maxlen):
        super(Embedder, self).__init__()
        self.name = "Embedder"  # To distinguish from the Bert embedder

        self.generator  = generator
        self.tokenizer  = tokenizer
        self.num_epochs = 0
        self.maxlen     = maxlen
        self.optimizer  = None

    @property
    def device(self):
        return next(self.parameters()).device

    def set_num_epochs(self, num):
        self.num_epochs = num
        
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
        
    def number_of_parameters(self):
        return sum(p.numel() for p in self.parameters())

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

class VectorEmbedder(Embedder):
    def __init__(self, tokenizer, hidden_size, vocab_size, maxlen=None):
        super(VectorEmbedder, self).__init__(VectorGenerator(hidden_size, vocab_size), tokenizer, maxlen)


class MatrixEmbedder(Embedder):
    def __init__(self, tokenizer, shape, vocab_size, maxlen=None):
        super(MatrixEmbedder, self).__init__(MatrixGenerator(shape, vocab_size), tokenizer, maxlen)    