import numpy
import torch
from tqdm import tqdm
def concept_categorization(embedder, dataset_file='project/datasets/concept_categorization/WordNet Words.CAT'):
    part_of_speech_to_pos_integer = {}
    pos_integer_to_part_of_speech = {}
    word_to_pos_integer = {}
    with open(dataset_file) as my_wordnet_file:
        current_integer = -1
        for my_line in my_wordnet_file:
            if my_line[0] != '\t':
                current_integer += 1
                part_of_speech_to_pos_integer[my_line[:-1]] = current_integer
                pos_integer_to_part_of_speech[current_integer] = my_line[:-1]
            if my_line[0:2] == '\t\t':
                my_word = my_line.split(' ')[0][2:].lower()
                word_to_pos_integer[my_word] = current_integer
    word_pos_list = []
    for my_word in word_to_pos_integer:
        word_pos_list.append((my_word, word_to_pos_integer[my_word]))
    number_pos = len(part_of_speech_to_pos_integer)
    all_embeddings = None
    with torch.no_grad():
        if embedder.name == 'Bert':
            all_embeddings = torch.empty((len(embedder.tokenizer), embedder.hidden_size), device='cuda')
            for word, token in embedder.tokenizer.get_vocab():
                all_embeddings[token] = embedder.model.get_input_embeddings()(torch.tensor(token, device='cuda'))
        else:
            words = [''] * len(embedder.tokenizer)
            for word, token in tqdm(embedder.tokenizer.get_vocab()):
                words[token] = word
            tokenized = embedder.tokenizer(words, embedder.maxlen).to('cuda')
            pre_embeddings = embedder(tokenized)
            all_embeddings = embedder.generator.vectorize(pre_embeddings)[:, 0]
    word_to_index = {}
    for word, idx in tqdm(embedder.tokenizer.get_vocab()):
        word_to_index[word] = idx
    filtered_word_pos_list = []
    for my_pair in word_pos_list:
        if my_pair[0] in word_to_index:
            filtered_word_pos_list.append(my_pair)
    word_pos_list = filtered_word_pos_list
    train_fraction = 0.8
    batch_size = 2000
    number_epochs = 50
    learning_rate = 0.5
    my_linear = torch.nn.Linear(all_embeddings.shape[1], number_pos, device='cuda')
    my_loss = torch.nn.CrossEntropyLoss()
    my_optimizer = torch.optim.SGD([my_linear.weight], lr=learning_rate)
    numpy.random.seed(177)
    new_word_pos_list = word_pos_list.copy()
    numpy.random.shuffle(new_word_pos_list)
    number_training = int(train_fraction*len(word_pos_list))
    number_testing = len(word_pos_list) - number_training
    training_list = word_pos_list[:number_training]
    testing_list = word_pos_list[number_training:]
    for my_epoch in range(number_epochs):
        training_numpy = numpy.array(training_list)
        numpy.random.shuffle(training_numpy)
        for i in range(0, number_training, batch_size):
            batch_numpy = training_numpy[i:i+batch_size]
            index_list = []
            for word_string in batch_numpy[:, 0]:
                index_list.append(word_to_index[word_string])
            index_numpy = numpy.array(index_list)
            input_torch = all_embeddings[index_numpy]
            target_torch = torch.tensor(batch_numpy[:, 1].astype(numpy.int64), device='cuda')
            output_torch = my_linear(input_torch)
            loss_output_torch = my_loss(output_torch, target_torch)
            loss_output_torch.backward()
            my_optimizer.step()
    testing_numpy = numpy.array(testing_list)
    index_list = []
    for word_string in testing_numpy[:, 0]:
        index_list.append(word_to_index[word_string])
    index_numpy = numpy.array(index_list)
    input_torch = all_embeddings[index_numpy]
    target_torch = torch.tensor(testing_numpy[:, 1].astype(numpy.int64), device='cuda')
    output_torch = my_linear(input_torch)
    max_output_torch = torch.max(output_torch, 1)[1]
    return torch.sum(max_output_torch == target_torch) / len(max_output_torch)