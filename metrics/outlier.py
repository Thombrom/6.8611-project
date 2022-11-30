import copy
import random
from tqdm import tqdm
import torch
from torchmetrics.functional import pairwise_cosine_similarity


class OutlierGroup():
    def __init__(self, words, outlier):
        self.words = words
        self.outlier = outlier

    def __repr__(self):
        return f"<Group: {self.words + [self.outlier]}, outlier:{self.outlier} >"

    def __str__(self):
        return self.__repr__()


class OutlierDataset():
    def __init__(self, embedder, dataset='', numgroups=10000):
        # what is going on
        categories = []

        with open(dataset, "r") as file:
            f = file.readline()
            category_list = (f.split('\n')[0], [])
            f = file.readline()

            if embedder.name == "Bert":
                all_words = []
                for word, token in embedder.tokenizer.get_vocab():
                    all_words.append(word)

            while f != "":
                f_list = f.split(',')

                if len(f_list) == 1:
                    append_list = copy.deepcopy(category_list)
                    categories.append(append_list)
                    category_list = (f_list[0].split('\n')[0], [])

                else:
                    for word in f_list:
                        word = word.strip()
                        if " " not in word:
                            if embedder.name == 'Bert':
                                if word in all_words:
                                    category_list[1].append(word.split('\n')[0])
                            else:
                                if embedder.tokenizer(word).item()!=0:
                                    # temp = embedder.tokenizer(word)
                                    category_list[1].append(word.split('\n')[0])
                                    # print(category_list[1])


                f = file.readline()

        self.outlier_groups = []

        print("prelim cats:", categories)
        categories = [cat for cat in categories if len(cat[1])>=5]
        print(categories)

        num_categories = len(categories)
        for i in range(numgroups):
            choices = random.sample(range(num_categories), 2)
            similar_category = categories[choices[0]]
            outlier_category = categories[choices[1]]

            # print(similar_category[0], outlier_category[0])

            similar_group = [similar_category[1][i] for i in random.sample(range(len(similar_category[1])), min(len(similar_category[1]),8))]
            outlier = random.choice(outlier_category[1])

            self.outlier_groups.append(OutlierGroup(similar_group, outlier))

    def get_outlier_groups(self):
        return self.outlier_groups



def detect_outliers(embedder, datafile, numgroups=10000, print_bool=0):
    def distance(a, b):
        return torch.norm(torch.subtract(a,b))
    def cosine_similiarity(a,b):
        return torch.dot(a,b)/(torch.norm(a)*torch.norm(b))

    if print_bool:
        print("apple:",embedder.tokenizer('apple').item)
    correct = 0
    index = 0
    total = 0

    dataset = OutlierDataset(embedder, datafile, numgroups)
    outlier_groups = dataset.get_outlier_groups()

    all_embeddings = None
    if embedder.name == 'Bert':
        all_embeddings = torch.empty((len(embedder.tokenizer), embedder.hidden_size))
        for word, token in embedder.tokenizer.get_vocab():
            all_embeddings[token] = embedder.model.get_input_embeddings()(torch.tensor(token))
    else:
        words = [''] * len(embedder.tokenizer)
        for word, token in tqdm(embedder.tokenizer.get_vocab()):
            words[token] = word
        tokenized = embedder.tokenizer(words, embedder.maxlen)
        pre_embeddings = embedder(tokenized)
        all_embeddings = embedder.generator.vectorize(pre_embeddings)[:, 0]

    word_to_index = {}
    for word, idx in tqdm(embedder.tokenizer.get_vocab()):
        word_to_index[word] = idx

    for group in outlier_groups:

        index+=1

        if index%500==0:
            print(index,":", correct/(total+1))

        similarity_list = [(all_embeddings[word_to_index[i]],[]) for i in group.words]

        # a = all_embeddings[word_to_index[group.word1]]
        # if print_bool:
        #     print(group.word1, a)
        # b = all_embeddings[word_to_index[group.word2]]
        # if print_bool:
        #     print(group.word2, b)
        # c = all_embeddings[word_to_index[group.word3]]
        # if print_bool:
        #     print(group.word3, c)
        expected_token = all_embeddings[word_to_index[group.outlier]]
        # if print_bool:
        #     print(group.outlier, expected_token)

        similarity_list = similarity_list +  [(expected_token, [])]
        random.shuffle(similarity_list)
        if print_bool:
            print(group)
        for i in range(len(similarity_list)):
            for j in range(len(similarity_list)):
                if i != j:
                    w1 = torch.flatten(similarity_list[i][0])
                    w2 = torch.flatten(similarity_list[j][0])
                    # print("w1:",w1)
                    similarity = abs(distance(w1, w2).item())
                    # print("similarity:", similarity)
                    similarity_list[i][1].append(similarity)
                    similarity_list[j][1].append(similarity)
                    if print_bool:
                        print(w1,w2,similarity)

        avg_similarity_list = [(i[0], sum(i[1]) / len(i[1])) for i in similarity_list]
        if print_bool:
            print(avg_similarity_list)
        least_similar = sorted(avg_similarity_list, key=lambda x: x[1])[0][0]
        # least_similar = similarity_list[0][0]

        # print(least_similar, expected_token)

        # if least_similar == expected_token:
        if distance(least_similar, expected_token)==0:
            correct += 1
        total += 1

    print(correct / total)

    return correct / total

    # return datafile




# class Dummy():
#     def __init__(self,a):
#         self.a = a
#         self.name = 'notbert'
#
#     def tokenizer(self, p):
#         return torch.Tensor(1)
#
# d = OutlierDataset(embedder=Dummy(4), dataset='../datasets/category_dataset/category_dataset.txt',numgroups=20)
#
# a = d.get_outlier_groups()
# for i in a:
#     print(i)