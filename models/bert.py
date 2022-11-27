from . import VectorEmbedder
from transformers import DistilBertTokenizer, DistilBertModel
import torch

class BertTokenizer():
    def __init__(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    def __call__(self, text, pad):
        value = self.tokenizer(text, return_tensors='pt', padding=True, max_length=pad)['input_ids'] 
        return value

    def __len__(self):
        return len(self.tokenizer.get_vocab())
    
    def get_vocab(self):
        yield from self.tokenizer.get_vocab().items()

class BertEmbedder(VectorEmbedder):
    def __init__(self):

        self.vocab_size = 30522
        self.hidden_size = 768
        bert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        tokenizer = BertTokenizer()
        super(BertEmbedder, self).__init__(tokenizer, self.hidden_size, self.vocab_size)
        self.model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.name = "Bert"
        
    def forward(self, x):
        return self.model(input_ids=x, attention_mask=torch.ones(*x.shape)).last_hidden_state[:, 1:][:, :-1]
    
# Usage:
# model = BertEmbedder()