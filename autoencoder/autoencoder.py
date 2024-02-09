import torch
import torch.nn.functional as F
from torch import nn

class Autoencoder(nn.Module):
    def __init__(self, input_dimensions, first_layer_hidden_dimensions=8400, second_layer_hidden_dimensions=3440, latent_space_dimensions=2800):
        super().__init__()

        self.input_dimensions = input_dimensions

        # Encoder
        self.first_hidden_layer_encoder = nn.Linear(in_features=input_dimensions, out_features=first_layer_hidden_dimensions)
        self.second_hidden_layer_encoder = nn.Linear(in_features=first_layer_hidden_dimensions, out_features=second_layer_hidden_dimensions)
        self.bottleneck_layer = nn.Linear(in_features=second_layer_hidden_dimensions, out_features=latent_space_dimensions)
       

        # Decoder
        self.second_hidden_layer_decoder = nn.Linear(in_features=latent_space_dimensions, out_features=second_layer_hidden_dimensions)
        self.first_hidden_layer_decoder = nn.Linear(in_features=second_layer_hidden_dimensions, out_features=first_layer_hidden_dimensions)
        self.reconstruction_layer1 = nn.Linear(in_features=first_layer_hidden_dimensions, out_features=input_dimensions)


        self.relu_activation = nn.ReLU()

    def encode(self, x):
        x = self.relu_activation(self.first_hidden_layer_encoder(x))
        x = self.relu_activation(self.second_hidden_layer_encoder(x))
        x = self.relu_activation(self.bottleneck_layer(x))

        return x

    def decode(self, z):
        x = self.relu_activation(self.second_hidden_layer_decoder(z))
        x = self.relu_activation(self.first_hidden_layer_decoder(x))
        x = self.reconstruction_layer1(x)

        return x


    def forward(self, x):
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        return x_reconstructed