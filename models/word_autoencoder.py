import torch
from torch.optim import Adam
from . import MatrixEmbedder


class WordAutoencoderModel(MatrixEmbedder):
    def __init__(self, tokenizer, shape, vocab_size, maxlen):
        super(WordAutoencoderModel, self).__init__(tokenizer, shape, vocab_size, maxlen)
        self.shape = shape
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embeddings = torch.nn.Parameter(torch.randn(vocab_size, *shape))
        self.set_optimizer(Adam(self.parameters(), lr=1e-3))
        self.my_sequential = torch.nn.Sequential(
            torch.nn.Conv2d(self.maxlen, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            torch.nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            torch.nn.ReLU(inplace=True),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, self.maxlen, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            torch.nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.my_sequential(self.embeddings[x])
    def save(self, folder, name):
        path = os.sep.join([ folder, name ])
        state = {
            'state_dict': self.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epochs': self.num_epochs,
            'shape': self.shape,
            'vocab_size': self.vocab_size,
            'tokenizer': self.tokenizer.save(),
            'maxlen': self.maxlen
        }
        torch.save(state, path)
    def load(cls, tokenizer_cls, path):
        state = torch.load(path)
        tokenizer = tokenizer_cls.load(state['tokenizer'])
        model = cls(tokenizer, state['shape'], state['vocab_size'], state['maxlen'])
        model.optimizer.load_state_dict(state['optimizer'])
        model.num_epochs = state['epochs']
        model.load_state_dict(state['state_dict'])
        return model