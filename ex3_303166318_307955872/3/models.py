import torch
from torch import nn

ACTIVATION = nn.functional.softplus


class Vae(nn.Module):
    def __init__(self, input_size, hidden_size, z_size, model_name):
        super().__init__()
        self.name = model_name

        # Encoder
        self.enc1 = nn.Linear(input_size, hidden_size)
        self.enc2 = nn.Linear(hidden_size, hidden_size)

        self.mu = nn.Linear(hidden_size, z_size)
        self.sigma = nn.Linear(hidden_size, z_size)

        # Decoder
        self.dec1 = nn.Linear(z_size, hidden_size)
        self.dec2 = nn.Linear(hidden_size, hidden_size)
        self.dec3 = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        x = self.enc1(x)
        x = ACTIVATION(x)
        x = self.enc2(x)
        x = ACTIVATION(x)

        mu = self.mu(x)
        sigma = self.sigma(x)
        epsilon = torch.randn_like(sigma)

        x = mu + sigma * epsilon

        x = ACTIVATION(x)
        x = self.dec1(x)
        x = ACTIVATION(x)
        x = self.dec2(x)
        x = ACTIVATION(x)
        x = self.dec3(x)
        x = torch.sigmoid(x)

        return mu, sigma, x
