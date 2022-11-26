import re
import tqdm
import torch

class Tokenizer:
    UNK_TOKEN  = 0
    MASK_TOKEN = 1
    PAD_TOKEN  = 2
    SOS_TOKEN  = 3
    EOS_TOKEN  = 4

    def __init__(self, min_occur=10):
        self.word_to_token = {}
        self.token_to_word = {}
        self.word_count = {}
        self.vocab_size = 0
        self.min_occur = min_occur

        for token in ['<unk>', '<mask>', '<pad>', '<sos>', '<eos>']:
            self.word_to_token[token] = self.vocab_size
            self.token_to_word[self.vocab_size] = token
            self.vocab_size += 1

    def save(self):
        return {
            'word_to_token': self.word_to_token,
            'token_to_word': self.token_to_word,
            'word_count': self.word_count,
            'vocab_size': self.vocab_size,
            'min_occur': self.min_occur
        }
    
    @classmethod
    def load(cls, state):
        tokenizer = cls(state['min_occur'])
        tokenizer.word_to_token = state['word_to_token']
        tokenizer.token_to_word = state['token_to_word']
        tokenizer.word_count = state['word_count']
        tokenizer.vocab_size = state['vocab_size']
        return tokenizer
        
    def fit(self, corpus):
        for text in corpus:
            text = text.strip().lower()
            words = re.findall(r"[\w']+|[.,!?;]", text)
            for word in words:
                if word not in self.word_count:
                    self.word_count[word] = 0
                self.word_count[word] += 1

        for text in corpus:
            text = text.strip().lower()
            words = re.findall(r"[\w']+|[.,!?;]", text)
            for word in words:
                if self.word_count[word] < self.min_occur:
                    continue
                if word in self.word_to_token:
                    continue
                self.word_to_token[word] = self.vocab_size
                self.token_to_word[self.vocab_size] = word
                self.vocab_size += 1
    
    def tokenize(self, corpus, pad=None):
        if not isinstance(corpus, (list,)):
            corpus = [corpus]
        
        tokenized_corpus = []
        for text in corpus:
            text = text.strip().lower()
            words = re.findall(r"[\w']+|[.,!?;]", text)
            tokenized_text = []
            for word in words:
                if word not in self.word_to_token:
                    tokenized_text.append(Tokenizer.UNK_TOKEN)
                else:
                    tokenized_text.append(self.word_to_token[word])

            if pad:
                tokenized_text = tokenized_text + [Tokenizer.PAD_TOKEN] * (pad - len(tokenized_text))
            
            tokenized_corpus.append(tokenized_text)
        return torch.Tensor(tokenized_corpus).to(torch.int64)

    def __call__(self, *args, **kwargs):
        return self.tokenize(*args, **kwargs)
    
    def de_tokenize(self, tokenized_corpus):
        if not isinstance(tokenized_corpus, (list,)):
            tokenized_corpus = [tokenized_corpus]
        
        corpus = []
        for tokenized_text in tokenized_corpus:
            text = []
            for token in tokenized_text:
                if isinstance(token, torch.Tensor):
                    token = token.item()
                text.append(self.token_to_word[token])
            corpus.append(" ".join(text))
        return corpus

    def __len__(self):
        return len(self.word_to_token)

    def get_vocab(self):
        yield from self.word_to_token.items()