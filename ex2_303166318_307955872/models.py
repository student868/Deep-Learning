import torch
from torch import nn


class RNN(nn.Module):
    def __init__(self, rnn_type, dropout: float, input_size, hidden_size):
        super().__init__()
        self.name = rnn_type
        if dropout > 0:
            self.name += ' with dropout ' + str(dropout)
        self.embedding = None
        self.RNN = rnn_type(num_layers=2)
        self.fc = None

    def forward(self, sentence):
        h0 = torch.zeros()
        if self.RNN == nn.LSTM:
            c0 = torch.zeros()

        x = self.embedding(sentence)
        x = self.RNN()
        return x
