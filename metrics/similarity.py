import re
import tqdm
import torch

class SimilarityPair():
    def __init__(self, first, second, score):
        self.first = first
        self.second = second
        self.score = score

    def __repr__(self):
        return f"<SimilarityPair .first={self.first}, .second={self.second}, .score={self.score} >"

    def __str__(self):
        return self.__repr__()

class MenDataset():
    def __init__(self, file="/content/project/datasets/MEN/MEN_dataset_natural_form_full"):
        similarity_pairs = []
        sim_regex = re.compile(r'(?P<first>\w+) (?P<second>\w+) (?P<score>\d+\.\d+)')
        with open(file) as f:
            for line in f:
                match = sim_regex.search(line)
                if match:
                    similarity_pairs.append(SimilarityPair(match.group("first"), match.group("second"), match.group("score")))
        
        self.pairs = similarity_pairs

def cosine_similarity(a, b):
    return torch.dot(a, b) / (torch.norm(a) * torch.norm(b))

def word_similarity(embedder, max_count=None, dataset_file="/content/project/datasets/MEN/MEN_dataset_natural_form_full"):
    dataset = MenDataset(dataset_file)
    
    raw_score_to_similarity = []
    score_to_count = {}
    
    for index, pair in tqdm.tqdm(enumerate(dataset.pairs)):
        a_tokens = embedder.tokenizer(pair.first)
        b_tokens = embedder.tokenizer(pair.second)
        
        a = embedder(a_tokens).squeeze()
        b = embedder(b_tokens).squeeze()

        a = embedder.generator.vectorize(a)
        b = embedder.generator.vectorize(b)

        if (a.dim() > 1 or b.dim() > 1):
            continue
        raw_score_to_similarity.append((pair.score, cosine_similarity(a, b).item()))
        
        if pair.score not in score_to_count:
            score_to_count[pair.score] = 0
        score_to_count[pair.score] += 1
        
        # For now because this damn thing is using 
        # up all the memory of my system with BERT
        if max_count and index > max_count:
            break
    
    # Now count the number of out of place pairs
    raw_score_to_similarity.sort(reverse=True, key=lambda x: x[1])
    out_of_place = 0
    
    #print(raw_score_to_similarity)
    
    for index, pair in tqdm.tqdm(enumerate(raw_score_to_similarity)):
        out_of_place += sum([ 1 if value[0] < pair[0] else 0 for value in raw_score_to_similarity[:index]])
        #print(sum([ 1 if value[0] < pair[0] else 0 for value in raw_score_to_similarity[:index]]))
        
    # Normalize and invert
    return 1 - out_of_place / ((len(raw_score_to_similarity)) * (len(raw_score_to_similarity) - 1) / 2)