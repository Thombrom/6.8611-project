import torch
import torch.nn as nn
import os
import tqdm
from torch.optim import Adam
from . import MatrixEmbedder

class NaiveMatrixModel(MatrixEmbedder):
    def __init__(self, tokenizer, shape, vocab_size):
        super(NaiveMatrixModel, self).__init__(tokenizer, shape, vocab_size)
        self.shape = shape
        self.vocab_size = vocab_size
        self.embeddings = nn.Parameter(torch.randn(vocab_size, *shape))

        self.set_optimizer(Adam(self.parameters(), lr=1e-5))
        self.name = "NaiveMatrixModel"
    def forward(self, x):
        return self.embeddings[x]
    
    def save(self, folder, name):
        path = os.sep.join([ folder, name ])
        state = {
            'state_dict':   self.state_dict(),
            'optimizer':    self.optimizer.state_dict(),
            'epochs':       self.num_epochs,
            'shape':        self.shape,
            'vocab_size':   self.vocab_size,
            'tokenizer':    self.tokenizer.save(),
            'maxlen':       self.maxlen
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
    
    def do_train(self, dataloader, epochs, savepath, epoch_func=lambda model: None):
        loss_func = nn.NLLLoss(reduction="sum", ignore_index=self.tokenizer.PAD_TOKEN)
        while self.num_epochs <= epochs:
            
            total_loss = 0
            
            for tokens in tqdm.tqdm(dataloader, position=0, leave=True):
                tokens = tokens.to(self.device)
                #mask_idx =  mask_idx.to(self.device)
                #replaced_token = replaced_token.to(self.device)

                output = self.forward(tokens)
                predictions = self.generator.predict(output)
                
                loss = loss_func(
                    predictions.contiguous().view(-1, predictions.size(-1)), 
                    tokens.contiguous().view(-1))
                loss.backward()          
                self.optimizer.step()
                self.optimizer.zero_grad()
                total_loss += loss
                
            print(f"Epoch {self.num_epochs}: Total loss {total_loss}")
            epoch_func(self)
            self.num_epochs += 1
            
            if savepath:
                self.save(savepath, f"{type(self).__name__}_{self.num_epochs}.tar")

    def get_all_embeddings(self):
        return self.embeddings.view(self.vocab_size, -1)