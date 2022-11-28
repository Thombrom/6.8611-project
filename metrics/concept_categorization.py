import numpy
import torch
def concept_categorization(embedder, dataset_file='WordNet Words.CAT'):
    part_of_speech_to_pos_integer = {}
    pos_integer_to_part_of_speech = {}
    word_integer_to_pos_integer = {}
    with open(dataset_file) as my_wordnet_file:
        current_integer = -1
        for my_line in my_wordnet_file:
            if my_line[0] != '\t':
                current_integer += 1
                part_of_speech_to_pos_integer[my_line[:-1]] = current_integer
                pos_integer_to_part_of_speech[current_integer] = my_line[:-1]
            if my_line[0:2] == '\t\t':
                my_word = my_line.split(' ')[0][2:].lower()
                if my_word in tokenizer.word_to_token:
                    word_integer_to_pos_integer[tokenizer.word_to_token[my_word]] = current_integer
    word_pos_list = []
    for my_word_integer in word_integer_to_pos_integer:
        word_pos_list.append((my_word_integer, word_integer_to_pos_integer[my_word_integer]))
    number_pos = len(part_of_speech_to_pos_integer)
    train_fraction=0.8
    batch_size=2000
    number_epochs=50
    if isinstance(embedder, NaiveMatrixModel) or isinstance(embedder, DoubleMatrixModel):
        my_embeddings = embedder.embeddings
        numpy.random.seed(177)
        new_word_pos_list = word_pos_list.copy()
        numpy.random.shuffle(new_word_pos_list)
        number_training = int(train_fraction*len(word_pos_list))
        number_testing = len(word_pos_list) - number_training
        training_list = word_pos_list[:number_training]
        testing_list = word_pos_list[number_training:]
        my_linear = torch.nn.Linear(my_embeddings.shape[1]*my_embeddings.shape[2], number_pos, device='cuda')
        my_loss = torch.nn.CrossEntropyLoss()
        my_optimizer = torch.optim.SGD([my_linear.weight], lr=0.5)
        for my_epoch in range(number_epochs):
            training_numpy = numpy.array(training_list)
            numpy.random.shuffle(training_numpy)
            for i in range(0, number_training, batch_size):
                batch_numpy = training_numpy[i:i+batch_size]
                torch_list = []
                for j in range(len(batch_numpy)):
                    torch_list.append(my_embeddings[batch_numpy[j, 0]])
                batch_torch = torch.stack(torch_list)
                batch_torch_2d = torch.reshape(batch_torch, shape=(batch_torch.shape[0], -1))
                input_torch = batch_torch_2d.detach()
                target_torch = torch.tensor(batch_numpy[:, 1], device='cuda')
                my_optimizer.zero_grad()
                linear_output_torch = my_linear(input_torch)
                loss_output_torch = my_loss(linear_output_torch, target_torch)
                loss_output_torch.backward()
                my_optimizer.step()
            testing_numpy = numpy.array(testing_list)
            torch_list = []
            for j in range(len(testing_numpy)):
                torch_list.append(my_embeddings[testing_numpy[j, 0]])
            batch_torch = torch.stack(torch_list)
            batch_torch_2d = torch.reshape(batch_torch, shape=(batch_torch.shape[0], -1))
            input_torch = batch_torch_2d.detach()
            target_torch = torch.tensor(testing_numpy[:, 1], device='cuda')
            linear_output_torch = my_linear(input_torch)
            max_output_torch = torch.max(linear_output_torch, 1)[1]
        return torch.sum(max_output_torch == target_torch) / len(max_output_torch)
    if isinstance(embedder, NaiveVectorModel):
        my_embeddings = embedder.embeddings
        numpy.random.seed(177)
        new_word_pos_list = word_pos_list.copy()
        numpy.random.shuffle(new_word_pos_list)
        number_training = int(train_fraction*len(word_pos_list))
        number_testing = len(word_pos_list) - number_training
        training_list = word_pos_list[:number_training]
        testing_list = word_pos_list[number_training:]
        my_linear = torch.nn.Linear(my_embeddings.weight.shape[1], number_pos, device='cuda')
        my_loss = torch.nn.CrossEntropyLoss()
        my_optimizer = torch.optim.SGD([my_linear.weight], lr=0.5)
        for my_epoch in range(number_epochs):
            training_numpy = numpy.array(training_list)
            numpy.random.shuffle(training_numpy)
            for i in range(0, number_training, batch_size):
                batch_numpy = training_numpy[i:i+batch_size]
                torch_list = []
                for j in range(len(batch_numpy)):
                    torch_list.append(my_embeddings.weight[batch_numpy[j, 0]])
                batch_torch_2d = torch.stack(torch_list)
                input_torch = batch_torch_2d.detach()
                target_torch = torch.tensor(batch_numpy[:, 1], device='cuda')
                my_optimizer.zero_grad()
                linear_output_torch = my_linear(input_torch)
                loss_output_torch = my_loss(linear_output_torch, target_torch)
                loss_output_torch.backward()
                my_optimizer.step()
            testing_numpy = numpy.array(testing_list)
            torch_list = []
            for j in range(len(testing_numpy)):
                torch_list.append(my_embeddings.weight[testing_numpy[j, 0]])
            batch_torch_2d = torch.stack(torch_list)
            input_torch = batch_torch_2d.detach()
            target_torch = torch.tensor(testing_numpy[:, 1], device='cuda')
            linear_output_torch = my_linear(input_torch)
            max_output_torch = torch.max(linear_output_torch, 1)[1]
        return torch.sum(max_output_torch == target_torch) / len(max_output_torch)
    if isinstance(embedder, BertEmbedder):
        my_embeddings = embedder.model.embeddings.word_embeddings.weight
        my_embeddings = my_embeddings.to('cuda')
        numpy.random.seed(177)
        new_word_pos_list = word_pos_list.copy()
        numpy.random.shuffle(new_word_pos_list)
        number_training = int(train_fraction*len(word_pos_list))
        number_testing = len(word_pos_list) - number_training
        training_list = word_pos_list[:number_training]
        testing_list = word_pos_list[number_training:]
        my_linear = torch.nn.Linear(my_embeddings.shape[1], number_pos, device='cuda')
        my_loss = torch.nn.CrossEntropyLoss()
        my_optimizer = torch.optim.SGD([my_linear.weight], lr=0.5)
        for my_epoch in range(number_epochs):
            training_numpy = numpy.array(training_list)
            numpy.random.shuffle(training_numpy)
            for i in range(0, number_training, batch_size):
                batch_numpy = training_numpy[i:i+batch_size]
                torch_list = []
                for j in range(len(batch_numpy)):
                    torch_list.append(my_embeddings[batch_numpy[j, 0]])
                batch_torch_2d = torch.stack(torch_list)
                input_torch = batch_torch_2d.detach()
                target_torch = torch.tensor(batch_numpy[:, 1], device='cuda')
                my_optimizer.zero_grad()
                linear_output_torch = my_linear(input_torch)
                loss_output_torch = my_loss(linear_output_torch, target_torch)
                loss_output_torch.backward()
                my_optimizer.step()
            testing_numpy = numpy.array(testing_list)
            torch_list = []
            for j in range(len(testing_numpy)):
                torch_list.append(my_embeddings[testing_numpy[j, 0]])
            batch_torch_2d = torch.stack(torch_list)
            input_torch = batch_torch_2d.detach()
            target_torch = torch.tensor(testing_numpy[:, 1], device='cuda')
            linear_output_torch = my_linear(input_torch)
            max_output_torch = torch.max(linear_output_torch, 1)[1]
        return torch.sum(max_output_torch == target_torch) / len(max_output_torch)
    print('Unsupported type')