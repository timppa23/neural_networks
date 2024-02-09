import torch
import torch.nn.functional as F
from torch import nn

class Variational2DAutoEncoder(nn.Module):
    def __init__(self, input_dimensions, hidden_dimensions1=8400, hidden_dimensions2=5400, hidden_dimensions3=3440, latent_space_dimensions=2800):
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
        self.out_feats = out_feats2
        self.input_dimensions = input_dimensions
        # print(f"out_feats2{out_feats2}")
        # (out_channels * out_feats2 * out_feats2)

        # Encoder
        #self.convolutional_layer1 = nn.Conv2d(in_channels=1, out_channels=middle_channels, kernel_size=conv1_params[0], stride=conv1_params[1], padding=conv1_params[2])
        #self.convolutional_layer2 = nn.Conv2d(in_channels=middle_channels, out_channels=out_channels,  kernel_size=conv2_params[0], stride=conv2_params[1], padding=conv2_params[2])
        self.hidden_layer_encoder1 = nn.Linear(in_features=input_dimensions, out_features=hidden_dimensions1)
        self.hidden_layer_encoder2 = nn.Linear(in_features=hidden_dimensions1, out_features=hidden_dimensions2)
        self.hidden_layer_encoder3 = nn.Linear(in_features=hidden_dimensions2, out_features=hidden_dimensions3)
        self.mu_layer = nn.Linear(in_features=hidden_dimensions3, out_features=latent_space_dimensions)
        self.sigma_layer = nn.Linear(in_features=hidden_dimensions3, out_features=latent_space_dimensions)

        # Decoder
        self.hidden_layer_decoder = nn.Linear(in_features=latent_space_dimensions, out_features=hidden_dimensions3)
        self.reconstruction_layer1 = nn.Linear(in_features=hidden_dimensions3, out_features=hidden_dimensions2)
        self.reconstruction_layer2 = nn.Linear(in_features=hidden_dimensions2, out_features=hidden_dimensions1)
        self.reconstruction_layer3 = nn.Linear(in_features=hidden_dimensions1, out_features=input_dimensions)
        #self.deconvolutional_layer1 = nn.ConvTranspose2d(in_channels=out_channels, out_channels=middle_channels,  kernel_size=conv2_params[0], stride=conv2_params[1], padding=conv2_params[2])
        #self.deconvolutional_layer2 = nn.ConvTranspose2d(in_channels=middle_channels, out_channels=1,  kernel_size=conv1_params[0], stride=conv1_params[1], padding=conv1_params[2])

        self.relu_activation = nn.ReLU()

    def encode(self, x):
        # Assuming input is a 2D matrix 
        # x = x.view(-1, 1, self.input_dimensions, self.input_dimensions)
        #x = self.relu_activation(self.convolutional_layer1(x))
        #x = self.relu_activation(self.convolutional_layer2(x))
        #x = x.view(x.size(0), -1)
        x = self.relu_activation(self.hidden_layer_encoder1(x))
        x = self.relu_activation(self.hidden_layer_encoder2(x))
        x = self.relu_activation(self.hidden_layer_encoder3(x))
        mu = self.mu_layer(x)
        sigma = self.sigma_layer(x)

        return mu, sigma

    def decode(self, z):
        x = self.relu_activation(self.hidden_layer_decoder(z))
        x = self.relu_activation(self.reconstruction_layer1(x))
        x = self.relu_activation(self.reconstruction_layer2(x))
        #x = self.relu_activation(self.reconstruction_layer3(x))
        #x = x.view(x.size(0), self.out_channels, self.out_feats, self.out_feats)  # Adjusted based on the encoder
        #x = self.relu_activation(self.deconvolutional_layer1(x))
        #x_reconstructed = self.deconvolutional_layer2(x)
        x_reconstructed = torch.sigmoid(self.reconstruction_layer3(x))
        return x_reconstructed

    def reparameterize(self, mu, sigma):
        # standard_deviation = torch.exp(0.5 * logvariance)
        epsilon = torch.randn_like(sigma)
        return mu + epsilon * sigma


    def forward(self, x):
        mu, sigma = self.encode(x)
        z_reparametrize = self.reparameterize(mu, sigma) 
        x_reconstructed = self.decode(z_reparametrize)
        return x_reconstructed, mu, sigma