import re
import os
import copy
import random
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

        categories = []

        # total_embeddings = embedder.get_all_embeddings()

        with open(dataset, "r") as file:
            f = file.readline()
            category_list = (f.split('\n')[0],[])
            f = file.readline()
            while f != "":
                f_list = f.split(',')

                if len(f_list)==1:
                    append_list = copy.deepcopy(category_list)
                    categories.append(append_list)
                    category_list = (f_list[0].split('\n')[0],[])

                else:
                    for word in f_list:
                        word = word.strip()
                        if " " not in word:
                            try:
                                temp = embedder.tokenizer(word)
                                category_list[1].append(word.split('\n')[0])
                                # print(category_list[1])
                            except:
                                continue

                f = file.readline()

        self.outlier_groups = []

        print(categories)

        num_categories = len(categories)
        for i in range(numgroups):
            choices = random.sample(range(num_categories),2)
            similar_category = categories[choices[0]]
            outlier_category = categories[choices[1]]

            # print(similar_category[0], outlier_category[0])


            similar_group = [similar_category[1][i] for i in random.sample(range(len(similar_category[1])),3)]
            outlier_group = similar_group + [random.choice(outlier_category[1])]

            # print(outlier_group)

            self.outlier_groups.append(OutlierGroup(*outlier_group))

    def get_outlier_groups(self):
        return self.outlier_groups


def detect_outliers(embedder, datafile):

    correct = 0
    total = 0

    dataset = OutlierDataset(embedder, datafile)
    # all_embeddings = embedder.get_all_embeddings()
    outlier_groups = dataset.get_outlier_groups()

    for group in outlier_groups:
        expected_token = embedder.tokenizer(group.outlier)
        outlier = embedder(expected_token).squeeze()
        outlier = embedder.generator.vectorize(outlier)

        word1_tokens = embedder.tokenizer(group.word1)
        word2_tokens = embedder.tokenizer(group.word2)
        word3_tokens = embedder.tokenizer(group.word3)

        word1 = embedder(word1_tokens).squeeze()
        word1 = embedder.generator.vectorize(word1)
        word2 = embedder(word2_tokens).squeeze()
        word2 = embedder.generator.vectorize(word2)
        word3 = embedder(word3_tokens).squeeze()
        word3 = embedder.generator.vectorize(word3)

        similiarity_list = [(word1,[]), (word2,[]), (word3,[]), (outlier,[])]
        random.shuffle(similiarity_list)

        for i in range(len(similiarity_list)):
            for j in range(len(similiarity_list)):
                if i!=j:
                    a = similiarity_list[i][0]
                    b = similiarity_list[j][0]
                    similarity = pairwise_cosine_similarity(a,b)
                    similiarity_list[i][1].append(similarity)
                    similiarity_list[j][1].append(similarity)

        avg_similarity_list = [(i[0], sum(i[1])/3) for i in similiarity_list]
        least_similar = sorted(avg_similarity_list, key=lambda x: x[1])[0]

        if least_similar == expected_token:
            correct += 1
        total +=1

    return correct/total

class Dummy():
    def __init__(self,a):
        self.a = a

    def tokenizer(self, p):
        return p

d = OutlierDataset(embedder=Dummy(4), dataset='../datasets/category_dataset/category_dataset.txt')

print(d.get_outlier_groups())















