import torch
import torch.nn.functional as F
from torch import nn

class Variational2DAutoEncoder(nn.Module):
    def __init__(self, input_dimensions, hidden_dimensions=200, latent_space_dimensions=20):
        super().__init__()

        self.input_dimensions = input_dimensions

        # Encoder
        self.convolutional_layer1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.convolutional_layer2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.hidden_layer_encoder = nn.Linear(in_features=(32 * input_dimensions * input_dimensions), out_features=hidden_dimensions)
        self.mean_layer = nn.Linear(in_features=hidden_dimensions, out_features=latent_space_dimensions)
        self.logvariance_layer = nn.Linear(in_features=hidden_dimensions, out_features=latent_space_dimensions)

        # Decoder
        self.hidden_layer_decoder = nn.Linear(in_features=latent_space_dimensions, out_features=hidden_dimensions)
        self.reconstruction_layer1 = nn.Linear(in_features=hidden_dimensions, out_features=32 * (input_dimensions * input_dimensions))
        self.deconvolutional_layer1 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.deconvolutional_layer2 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)

        self.relu_activation = nn.ReLU()

    def encode(self, x):
        # Assuming input is a 2D matrix 
        # x = x.view(-1, 1, self.input_dimensions, self.input_dimensions)
        #print(f"x encoder shape: {x.shape}")
        x = self.relu_activation(self.convolutional_layer1(x))
        x = self.relu_activation(self.convolutional_layer2(x))
        x = x.view(x.size(0), -1)
        x = self.relu_activation(self.hidden_layer_encoder(x))
        mean = self.mean_layer(x)
        logvariance = self.logvariance_layer(x)
        return mean, logvariance

    def decode(self, z):
        x = self.relu_activation(self.hidden_layer_decoder(z))
        x = self.relu_activation(self.reconstruction_layer1(x))
        #print(f"x decoder1 shape: {x.shape}")
        x = x.view(x.size(0), 32, (self.input_dimensions), (self.input_dimensions))  # Adjusted based on the encoder
        #print(f"x decoder2 shape: {x.shape}")
        x = self.relu_activation(self.deconvolutional_layer1(x))
        x_reconstructed = torch.sigmoid(self.deconvolutional_layer2(x))
        return x_reconstructed.view(x_reconstructed.size(0), -1)

    def reparameterize(self, mean, logvariance):
        standard_deviation = torch.exp(0.5 * logvariance)
        epsilon = torch.randn_like(standard_deviation)
        return mean + epsilon * standard_deviation


    def forward(self, x):
        mean, logvariance = self.encode(x)
        z_reparametrize = self.reparameterize(mean, logvariance) 
        x_reconstructed = self.decode(z_reparametrize)
        return x_reconstructed, mean, logvariance

# if __name__ == "__main__":
#     x = torch.randn(4, 420 * 420)  # Assuming input is a 2D matrix of size (420, 420)
#     vae = Variational2DAutoEncoder(input_dimensions=420 * 420, hidden_dimensions=200, latent_space_dimensions=20)
#     x_reconstructed, mean, logvariance = vae(x)
#     print(x_reconstructed.shape)
#     print(mean.shape)
#     print(logvariance.shape)