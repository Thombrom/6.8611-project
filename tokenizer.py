import re

class Tokenizer:
    def __init__(self, min_occur=10):
        self.word_to_token = {}
        self.token_to_word = {}
        self.word_count = {}
        self.vocab_size = 0
        self.min_occur = min_occur

        for token in ['<unk>', '<pad>', '<sos>', '<eos>']:
            self.word_to_token[token] = self.vocab_size
            self.token_to_word[self.vocab_size] = token
            self.vocab_size += 1

    

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

    def tokenize(self, corpus):
        tokenized_corpus = []
        for text in corpus:
            text = text.strip().lower()
            words = re.findall(r"[\w']+|[.,!?;]", text)
            tokenized_text = []
            for word in words:
                if word not in self.word_to_token:
                    tokenized_text.append(0)
                else:
                    tokenized_text.append(self.word_to_token[word])
            tokenized_corpus.append(tokenized_text)
        return tokenized_corpus

    def de_tokenize(self, tokenized_corpus):
        corpus = []
        for tokenized_text in tokenized_corpus:
            text = []
            for token in tokenized_text:
                text.append(self.token_to_word[token])
            corpus.append(" ".join(text))
        return corpus