class WikicorpDataset:
    def __init__(self, file="datasets/Wikicorp/data"):
        self.file = open(file, "r")
    
    def __del__(self):
        close(self.file)

    def __next__(self):
        line = next(self.file)
        lines = []
        while line:
            lines.append(line)
            if (line == '---END.OF.DOCUMENT---'):
                return WikicorpArticle(lines[1], lines[2:])

class WikicorpArticle:
    def __init__(self, title, text):
        self.title = title
        self.text = text
