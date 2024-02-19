import torch
import torch.nn.functional as F
from torch import nn

class ConvoAutoencoder(nn.Module):
    def __init__(self, input_dimensions, first_layer_hidden_dimensions=8400, second_layer_hidden_dimensions=3440, latent_space_dimensions=2800):
        super().__init__()

        middle_channels = 2
        out_channels = 4
        conv1_params = [5,2,0]
        conv2_params = [5,2,0]
        K = conv1_params[0]
        I = input_dimensions
        P = conv1_params[2]
        S = conv1_params[1]
        out_feats1 = (I - K + 2 * P) // S + 1
        out_feats2 = (out_feats1 - K + 2 * P) // S + 1

        self.out_channels = out_channels
        self.out_feats1 = out_feats1
        convo_out_feats = out_feats1 * out_feats1 * 4
        self.out_feats2 = out_feats2
        self.input_dimensions = input_dimensions
        # print(f"out_feats2{out_feats2}")
        # (out_channels * out_feats2 * out_feats2)

        self.input_dimensions = input_dimensions

        # Encoder
        self.convolutional_layer1 = nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=conv1_params[0], stride=conv1_params[1], padding=conv1_params[2])
        self.first_hidden_layer_encoder = nn.Linear(in_features=convo_out_feats, out_features=first_layer_hidden_dimensions)
        self.second_hidden_layer_encoder = nn.Linear(in_features=first_layer_hidden_dimensions, out_features=second_layer_hidden_dimensions)
        self.bottleneck_layer = nn.Linear(in_features=second_layer_hidden_dimensions, out_features=latent_space_dimensions)
       

        # Decoder
        self.second_hidden_layer_decoder = nn.Linear(in_features=latent_space_dimensions, out_features=second_layer_hidden_dimensions)
        self.first_hidden_layer_decoder = nn.Linear(in_features=second_layer_hidden_dimensions, out_features=first_layer_hidden_dimensions)
        self.reconstruction_layer1 = nn.Linear(in_features=first_layer_hidden_dimensions, out_features=convo_out_feats)
        self.deconvolutional_layer1 = nn.ConvTranspose2d(in_channels=out_channels, out_channels=1,  kernel_size=conv1_params[0], stride=conv1_params[1], padding=conv1_params[2] , output_padding=1)


        self.relu_activation = nn.ReLU()

    def encode(self, x):
        x = self.relu_activation(self.convolutional_layer1(x))
        x = x.view(x.size(0), -1)
        x = self.relu_activation(self.first_hidden_layer_encoder(x))
        x = self.relu_activation(self.second_hidden_layer_encoder(x))
        x = self.relu_activation(self.bottleneck_layer(x))

        return x

    def decode(self, z):
        x = self.relu_activation(self.second_hidden_layer_decoder(z))
        x = self.relu_activation(self.first_hidden_layer_decoder(x))
        x = self.relu_activation(self.reconstruction_layer1(x))
        x = x.view(x.size(0), self.out_channels, self.out_feats1, self.out_feats1)
        x = self.deconvolutional_layer1(x)

        return x


    def forward(self, x):
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        return x_reconstructed