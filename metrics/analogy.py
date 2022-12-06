import re
from tqdm import tqdm
import torch
import sklearn.metrics.pairwise
from torchmetrics.functional import pairwise_cosine_similarity

class Analogy():
    def __init__(self, word_1, analogy_1, word_2, analogy_2):
        self.word_1 = word_1
        self.analogy_1 = analogy_1
        self.word_2 = word_2
        self.analogy_2 = analogy_2

    def __repr__(self):
        return f"<AnalogyPair {self.word_1} -> {self.analogy_1} :: {self.word_2} -> {self.analogy_2}  >"

    def __str__(self):
        return self.__repr__()

    def get(self):
        return [self.word_1, self.analogy_1, self.word_2, self.analogy_2]

class AnalogyDataset():
    def __init__(self, dataset):
        self.dataset = dataset
        print("Creating analogies ..")
    def get_analogies(self):
        with open(self.dataset) as f:
            for line in tqdm(f):
                words = line.split()
                if len(words) == 4:
                    word_1, analogy_1, word_2, analogy_2 = words
                    yield Analogy(word_1, analogy_1, word_2, analogy_2)


def get_word_analogy_score(embedder, closest_k=5, dataset_file="project/datasets/word_analogy_dataset/questions-words.txt"):
    dataset = AnalogyDataset(dataset_file)

    # Speed things up for BERT as otherwise this is very slow
    all_embeddings = None
    with torch.no_grad():
        if embedder.name == 'Bert':
            all_embeddings = torch.empty((len(embedder.tokenizer), embedder.hidden_size), device=embedder.device)
            for word, token in embedder.tokenizer.get_vocab():
                all_embeddings[token] = embedder.model.get_input_embeddings()(torch.tensor(token, device=embedder.device))
        else:
            words = [''] * len(embedder.tokenizer)
            for word, token in tqdm(embedder.tokenizer.get_vocab()):
                words[token] = word
            tokenized = embedder.tokenizer(words, embedder.maxlen).to('cuda')
            pre_embeddings = embedder(tokenized).to(embedder.device)
            all_embeddings = embedder.generator.vectorize(pre_embeddings)[:, 0]
        
    
    word_to_index = {}
    for word, idx in tqdm(embedder.tokenizer.get_vocab()):
        word_to_index[word] = idx
    
    matching_tokens = 0
    total = 0
    index = 0
    for analogy_line in tqdm((dataset.get_analogies()), position=0, leave=True):
        index += 1
        word_1, analogy_1, word_2, analogy_2 = analogy_line.get()
        word_1 = word_1.lower()
        analogy_1 = analogy_1.lower()
        word_2 = word_2.lower()
        analogy_2 = analogy_2.lower()
        
        if (set([word_1, analogy_1, word_2, analogy_2]) - set(word_to_index.keys())):
            continue

        if index % 1000 == 0:
          print(f"accuracy {matching_tokens/(total+1)}")
        
        a = all_embeddings[word_to_index[word_1]]
        a_analogy = all_embeddings[word_to_index[analogy_1]]
        b = all_embeddings[word_to_index[word_2]]
        expected_token = word_to_index[analogy_2]

        analogy_embedding = a_analogy - a + b

        #get top k closest tokens
        top_analogy_tokens = torch.topk(pairwise_cosine_similarity(all_embeddings, analogy_embedding.unsqueeze(0)).squeeze(), closest_k)[1].tolist()

        if expected_token in set(top_analogy_tokens):
            matching_tokens += 1
        total +=1


    return matching_tokens / total
