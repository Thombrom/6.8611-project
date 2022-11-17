from torch.utils.data import Dataset, DataLoader
import re, random, tqdm, torch

class Wikicorp:
    def __init__(self, file="datasets/Wikicorp/data"):
        self.file = open(file, "r")
    
    def __del__(self):
        close(self.file)

    def __iter__(self):
        return self

    def __next__(self):
        next(self.file)
        line = next(self.file).strip()
        lines = []
        while line != '---END.OF.DOCUMENT---':
            lines.append(line)
            line = next(self.file).strip()

        return WikicorpArticle(lines[0], lines[1:])
    
    @staticmethod
    def create_dataset(articles, file):
        with open(file, "w") as f:
            for article in articles:
                f.write("\n{}\n".format(article.title))
                
                for line in article.text:
                    f.write("{}\n".format(line))
                
                f.write("---END.OF.DOCUMENT---\n")

class WikicorpArticle:
    def __init__(self, title, text):
        self.title = title
        self.text = text

class WikicorpDataset(Dataset):
    def __init__(self, tokenizer, file="datasets/Wikicorp/data", max_len=None, pad=True):
        self.sentences = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad = pad
        
        for article in tqdm.tqdm(Wikicorp(file=file), position=0, leave=True):
            sentences = [ sentence for sentence in re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", " ".join(article.text)) ]
            if max_len:
                self.sentences.extend([ sentence for sentence in sentences if len(tokenizer.tokenize(sentence)[0]) <= max_len ])
            else:
                self.sentences.extend(sentences)
            # This might be possible to speed up
            
            
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        tokens = self.tokenizer.tokenize(sentence)[0]
        if self.max_len:
            assert len(tokens) <= self.max_len 
        
        mask_idx = random.randint(0, len(tokens) - 1)
        replaced_token = tokens[mask_idx].item()
        tokens[mask_idx] = self.tokenizer.MASK_TOKEN
        
        if self.pad:
            tokens = tokens.tolist()
            tokens = torch.Tensor(tokens + [self.tokenizer.PAD_TOKEN] * (self.max_len - len(tokens))).to(torch.int64)
        
        return tokens, mask_idx, replaced_token    