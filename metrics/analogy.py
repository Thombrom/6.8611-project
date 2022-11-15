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
        self.analogies = []
        with open(dataset) as f:
            print("Creating analogies ..")
            for line in tqdm(f):
                words = line.split()
                if len(words) == 4:
                    word_1, analogy_1, word_2, analogy_2 = words
                    self.analogies.append(Analogy(word_1, analogy_1, word_2, analogy_2))


def get_word_analogy_score(embedder, dataset_file="/datasets/Word_analogy_dataset/questions-words.txt"):
    dataset = AnalogyDataset(dataset_file)
    (vocab_size, embedd_size) = embedder.vocab_size, embedder.hidden_size
    all_embeddings = embedder.embeddings
    
    matching_tokens = 0
    total = 0
    for analogy_line in tqdm((dataset.analogies)):
        word_1, analogy_1, word_2, analogy_2 = analogy_line.get()


        if embedder.name == "bert":
            try:
                expected_token = embedder.word_to_token.get(analogy_2)
                a_tokens = embedder.word_to_token.get(word_1)
                a_analogy_tokens = embedder.word_to_token[analogy_1]
                b_tokens = embedder.word_to_token[word_2]

                a = embedder.embeddings[a_tokens]
                a_analogy = embedder.embeddings[a_analogy_tokens]
                b = embedder.embeddings[b_tokens]

                a = embedder.generator.vectorize(a)
                a_analogy = embedder.generator.vectorize(a_analogy)
                b = embedder.generator.vectorize(b)
            except:
                continue
        else:
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

        print(f"token {analogy_token} expected token {expected_token}")
        if analogy_token == expected_token:
            matching_tokens += 1
        total +=1

    return matching_tokens / total



if __name__ == '__main__':
    pass
    
    # dataset = AnalogyDataset("/Users/hophinkibona/Desktop/6.8611-project/datasets/Word_analogy_dataset/questions-words.txt")

    # print("hey", len(dataset.analogies))
    # # all_words = torch.tensor([[1.0, 2.0], [2.0, 3.0], [1.0, 1.0]])

    # # a = torch.tensor(([[2.0, 3.0]]))


    # # print("all_words", all_words)
    # # print("a", a)

    # # # a_ = torch.rand(10)
    # # # b = torch.rand(10)

    # # print(pairwise_cosine_similarity(all_words, a).flatten())
