import re

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
    def __init__(self, file="datasets/MEN/MEN_dataset_natural_form_full"):
        
        similarity_pairs = []
        sim_regex = re.compile(r'(?P<first>\w+) (?P<second>\w+) (?P<score>\d+\.\d+)')
        with open(file) as f:
            for line in f:
                match = sim_regex.search(line)
                if match:
                    similarity_pairs.append(SimilarityPair(match.group("first"), match.group("second"), match.group("score")))
        
        self.similarity_pairs = similarity_pairs

def word_similarity(embedder, tokenizer):
    dataset = MenDataset()

    
