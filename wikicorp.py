class WikicorpDataset:
    def __init__(self, file="datasets/Wikicorp/data"):
        self.file = open(file, "r")
        next(self.file)
    
    def __del__(self):
        close(self.file)

    def __iter__(self):
        return self

    def __next__(self):
        line = next(self.file).strip()
        lines = [line]
        while line != '---END.OF.DOCUMENT---':
            lines.append(line)
            line = next(self.file).strip()

        next(self.file)
        return WikicorpArticle(lines[0], lines[1:])

class WikicorpArticle:
    def __init__(self, title, text):
        self.title = title
        self.text = text
