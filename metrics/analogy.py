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

class AnalogyDataset():
    def __init__(self, dataset):
        self.analogies = []
        sim_regex = re.compile(r'(?P<word_1>\w+) (?P<analogy_1>\w+) (?P<word_2>\w+) (?P<analogy_2>\w+) ')
        with open(dataset) as f:
            print("Creating analogies ..")
            for line in tqdm(f):
                match = sim_regex.search(line)
                if match:
                    self.analogies.append(Analogy(match.group("word_1"), match.group("analogy_1"), match.group("word_2")), match.group("analogy_2"))

# def cosine_similarity(a, b):
#     return torch.dot(a, b) / (torch.norm(a) * torch.norm(b))
def get_word_analogy_score(embedder, dataset_file="/datasets/Word_analogy_dataset/questions-words.txt"):
    dataset = AnalogyDataset(dataset_file)
    # (vocab_size, embedd_size)
    # vocab_size = embedder.vocab_size
    # embedd_size = embedder.hidden_size
    all_embeddings = embedder.embeddings()
    matching_tokens = 0
    for analogy_line in tqdm((dataset.analogies)):
        word_1, analogy_1, word_2, analogy_2 = analogy_line

        expected_token = embedder.tokenizer(analogy_2)

        a_tokens = embedder.tokenizer(word_1)
        a_analogy_tokens = embedder.tokenizer(analogy_1)
        b_tokens = embedder.tokenizer(word_2)

        a = embedder(a_tokens).squeeze()
        a_analogy = embedder(a_analogy_tokens).squeeze()
        b = embedder(b_tokens).squeeze()

        a = embedder.generator.vectorize(a)
        a_analogy = embedder.generator.vectorize(a_analogy)
        b = embedder.generator.vectorize(b)

        analogy_embedding = a_analogy - a + b
        analogy_token  = torch.argmax(pairwise_cosine_similarity(all_embeddings, analogy_embedding).squeeze())

        if analogy_token == expected_token:
            matching_tokens += 1

    return matching_tokens / embedder.vocab_size

# all_words = torch.rand((5, 10))

# a = torch.rand(((1, 10)))


# print("all_words", all_words)
# print("a", a)

# a_ = torch.rand(10)
# b = torch.rand(10)

# print(pairwise_cosine_similarity(all_words, a).flatten())

# if __name__ == '__main__':
#     model = BertEmbedder()
#     print(get_word_analogy_score(model))