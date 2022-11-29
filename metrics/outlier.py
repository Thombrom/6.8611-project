import copy
import random
from tqdm import tqdm
import torch
from torchmetrics.functional import pairwise_cosine_similarity


class OutlierGroup():
    def __init__(self, word1, word2, word3, outlier):
        self.word1 = word1
        self.word2 = word2
        self.word3 = word3
        self.outlier = outlier

    def __repr__(self):
        return f"<Group:[ {self.word1}, {self.word2}, {self.word3}, {self.outlier}* ]>"

    def __str__(self):
        return self.__repr__()


class OutlierDataset():
    def __init__(self, embedder, dataset='', numgroups=10000):
        self.outlier_groups = []
        # what is going on
        # categories = []
        #
        # with open(dataset, "r") as file:
        #     f = file.readline()
        #     category_list = (f.split('\n')[0], [])
        #     f = file.readline()
        #     while f != "":
        #         f_list = f.split(',')
        #
        #         if len(f_list) == 1:
        #             append_list = copy.deepcopy(category_list)
        #             categories.append(append_list)
        #             category_list = (f_list[0].split('\n')[0], [])
        #
        #         else:
        #             for word in f_list:
        #                 word = word.strip()
        #                 if " " not in word:
        #                     if 0 == 0:  # try:
        #                         # temp = embedder.tokenizer(word)
        #                         category_list[1].append(word.split('\n')[0])
        #                         # print(category_list[1])
        #                     # except:
        #                     #     continue
        #
        #         f = file.readline()
        #
        # self.outlier_groups = []
        #
        # print(categories)
        #
        # num_categories = len(categories)
        # for i in range(numgroups):
        #     choices = random.sample(range(num_categories), 2)
        #     similar_category = categories[choices[0]]
        #     outlier_category = categories[choices[1]]
        #
        #     # print(similar_category[0], outlier_category[0])
        #
        #     similar_group = [similar_category[1][i] for i in random.sample(range(len(similar_category[1])), 3)]
        #     outlier_group = similar_group + [random.choice(outlier_category[1])]
        #     random.shuffle(outlier_group)
        #
        #     # print(outlier_group)
        #
        #     self.outlier_groups.append(OutlierGroup(*outlier_group))

    def get_outlier_groups(self):
        return self.outlier_groups


def detect_outliers(a):
    return a

# def detect_outliers(datafile): #add embedder
    # print("apple:",embedder.tokenizer('apple').item)
    # correct = 0
    # test
    # test
    # test
    # total = 0

    # dataset = OutlierDataset(embedder, datafile)
    # outlier_groups = dataset.get_outlier_groups()
    #
    # all_embeddings = None
    # if embedder.name == 'Bert':
    #     all_embeddings = torch.empty((len(embedder.tokenizer), embedder.hidden_size))
    #     for word, token in embedder.tokenizer.get_vocab():
    #         all_embeddings[token] = embedder.model.get_input_embeddings()(torch.tensor(token))
    # else:
    #     words = [''] * len(embedder.tokenizer)
    #     for word, token in tqdm(embedder.tokenizer.get_vocab()):
    #         words[token] = word
    #     tokenized = embedder.tokenizer(words, embedder.maxlen)
    #     pre_embeddings = embedder(tokenized)
    #     all_embeddings = embedder.generator.vectorize(pre_embeddings)[:, 0]
    #
    # word_to_index = {}
    # for word, idx in tqdm(embedder.tokenizer.get_vocab()):
    #     word_to_index[word] = idx
    #
    # for group in outlier_groups:
    #
    #     a = all_embeddings[word_to_index[group.word1]]
    #     b = all_embeddings[word_to_index[group.word2]]
    #     c = all_embeddings[word_to_index[group.word3]]
    #     expected_token = all_embeddings[word_to_index[group.outlier]]
    #
    #     similarity_list = [(a, []), (b, []), (c, []), (expected_token, [])]
    #     random.shuffle(similarity_list)
    #
    #     for i in range(len(similarity_list)):
    #         for j in range(len(similarity_list)):
    #             if i != j:
    #                 w1 = similarity_list[i][0]
    #                 w2 = similarity_list[j][0]
    #                 similarity = pairwise_cosine_similarity(w1, w2)
    #                 similarity_list[i][1].append(similarity)
    #                 similarity_list[j][1].append(similarity)
    #
    #     avg_similarity_list = [(i[0], sum(i[1]) / 3) for i in similarity_list]
    #     least_similar = sorted(avg_similarity_list, key=lambda x: x[1])[0]
    #
    #     if least_similar == expected_token:
    #         correct += 1
    #     total += 1

    # return correct / total

    # return datafile

# class Dummy():
#     def __init__(self,a):
#         self.a = a
#
#     def tokenizer(self, p):
#         return p
#
# d = OutlierDataset(embedder=Dummy(4), dataset='../datasets/category_dataset/category_dataset.txt')
#
# print(d.get_outlier_groups())