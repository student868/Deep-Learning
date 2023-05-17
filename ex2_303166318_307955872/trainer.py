import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

# ---------------- TRAINING constants ----------------
BATCH_SIZE = 20

# ---------------- Model constants ----------------
HIDDEN_UNITS = 200
NUM_LAYERS = 2
MAX_GRAD_NORM = 5


# -------------------------------------------------

class RNN(nn.Module):
    def __init__(self, rnn_type, vocabulary_size: int, w_init=0.1, dropout: float = 0):
        super().__init__()

        self.name = rnn_type.__name__
        if dropout > 0:
            self.name += ' with dropout ' + str(dropout)

        self.embedding = nn.Embedding(vocabulary_size, HIDDEN_UNITS)
        self.rnn = rnn_type(input_size=HIDDEN_UNITS, hidden_size=HIDDEN_UNITS, num_layers=NUM_LAYERS, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(HIDDEN_UNITS, vocabulary_size)

        for param in self.parameters():
            nn.init.uniform_(param, -w_init, w_init)

    def forward(self, x, h):
        x = self.embedding(x)
        if self.rnn.mode == 'LSTM':
            h = [state.detach() for state in h]
            output, (h, c) = self.rnn(x, h)
            output = output.reshape(output.size(0) * output.size(1), output.size(2))
            y = self.fc(output)
            return y, (h, c)
        else:  # GRU
            h = h.detach()
            output, h = self.rnn(x, h)
            output = output.reshape(output.size(0) * output.size(1), output.size(2))
            y = self.fc(output)
            return y, h

    def initial_state(self, device):
        # initialize the hidden states to zero

        if self.rnn.mode == 'LSTM':
            return (torch.zeros(NUM_LAYERS, BATCH_SIZE, HIDDEN_UNITS).to(device),
                    torch.zeros(NUM_LAYERS, BATCH_SIZE, HIDDEN_UNITS).to(device))
        else:  # GRU
            return torch.zeros(NUM_LAYERS, BATCH_SIZE, HIDDEN_UNITS).to(device)


def train(device, model, train_data, loss_fn, optimizer, sequence_length):
    model.train()

    states = model.initial_state(device)
    # iterate over all training set
    for i in range(0, train_data.size(1) - sequence_length, sequence_length):
        # batch inputs and targets
        inputs = train_data[:, i:i + sequence_length].to(device)
        targets = train_data[:, (i + 1):(i + 1) + sequence_length].to(device)

        # forward
        outputs, states = model(inputs, states)
        loss = loss_fn(outputs, targets)

        # gradient step
        model.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()


def test(device, model, data, loss_fn, sequence_length):
    model.eval()

    # iterate over all dataset
    states = model.initial_state(device)
    losses = []
    with torch.no_grad():
        for i in range(0, data.size(1) - sequence_length, sequence_length):
            # batch inputs and targets
            inputs = data[:, i:i + sequence_length].to(device)
            targets = data[:, (i + 1):(i + 1) + sequence_length].to(device)

            # forward
            outputs, states = model(inputs, states)
            losses.append(loss_fn(outputs, targets) / BATCH_SIZE)

    return torch.exp(torch.mean(torch.Tensor(losses)))
