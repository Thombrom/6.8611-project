from models import VectorEmbedder
from transformers import DistilBertTokenizer, DistilBertModel
import torch

class BertEmbedder(VectorEmbedder):
    def __init__(self):
        bert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        tokenizer = lambda text: bert_tokenizer(text, return_tensors='pt')["input_ids"]
        super(BertEmbedder, self).__init__(tokenizer, 768, 30522)
        self.model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        
    def forward(self, x):
        return self.model(input_ids=x, attention_mask=torch.ones(*x.shape)).last_hidden_state[:, 1:][:, :-1]