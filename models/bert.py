from . import VectorEmbedder
from transformers import DistilBertTokenizer, DistilBertModel
import torch
class BertEmbedder(VectorEmbedder):
    def __init__(self):

        self.vocab_size = 30522
        self.hidden_size = 768
        bert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        tokenizer = lambda text: bert_tokenizer(text, return_tensors='pt')["input_ids"]
        super(BertEmbedder, self).__init__(tokenizer, self.hidden_size, self.vocab_size)
        self.model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.name = "bert"
        self.embeddings = torch.empty((self.vocab_size, self.hidden_size))

        self.word_to_token = {}
        self.token_to_word = {}
        for word, token in bert_tokenizer.get_vocab().items():
            self.embeddings[token] = self.model.get_input_embeddings()(torch.tensor(token))
            self.word_to_token[word] = token
            self.token_to_word[token] = word

        
    def forward(self, x):
        return self.model(input_ids=x, attention_mask=torch.ones(*x.shape)).last_hidden_state[:, 1:][:, :-1]

    