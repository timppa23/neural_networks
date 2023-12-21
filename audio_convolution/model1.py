import torch
import torch.nn.functional as F
from torch import nn

class Variational2DAutoEncoder(nn.Module):
    def __init__(self, input_dimensions, hidden_dimensions=200, latent_space_dimensions=20):
        super().__init__()

        # Encoder
        self.convolutional_layer1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)
        #self.convolutional_layer2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.hidden_layer_encoder = nn.Linear(in_features=(input_dimensions - 2) * (input_dimensions - 2), out_features=hidden_dimensions)
        self.mean_layer = nn.Linear(in_features=hidden_dimensions, out_features=latent_space_dimensions)
        self.logvariance_layer = nn.Linear(in_features=hidden_dimensions, out_features=latent_space_dimensions)

        # Decoder
        self.hidden_layer_decoder = nn.Linear(in_features=latent_space_dimensions, out_features=hidden_dimensions)
        self.reconstruction_layer1 = nn.Linear(in_features=hidden_dimensions, out_features=(input_dimensions - 2) * (input_dimensions - 2))
        #self.deconvolutional_layer1 = nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.deconvolutional_layer2 = nn.ConvTranspose2d(in_channels=4, out_channels=1, kernel_size=3, stride=1, padding=1)

        self.relu_activation = nn.ReLU()

    def encode(self, x):
        # Assuming input is a 2D matrix of size (420, 420)
        x = x.view(-1, 1, self.input_dimensions, self.input_dimensions)
        x = self.relu_activation(self.convolutional_layer1(x))
        # x = self.relu_activation(self.convolutional_layer2(x))  # Commented out
        x = x.view(x.size(0), -1)
        x = self.relu_activation(self.hidden_layer_encoder(x))
        mean = self.mean_layer(x)
        logvariance = self.logvariance_layer(x)
        return mean, logvariance

    def decode(self, z):
        x = self.relu_activation(self.hidden_layer_decoder(z))
        x = self.relu_activation(self.reconstruction_layer1(x))
        x = x.view(x.size(0), (self.input_dimensions - 2), (self.input_dimensions - 2))  # Adjusted based on the encoder
        # x = self.relu_activation(self.deconvolutional_layer1(x))  # Commented out
        x_reconstructed = torch.sigmoid(self.deconvolutional_layer2(x))
        return x_reconstructed.view(x_reconstructed.size(0), -1)

    def reparameterize(self, mean, logvariance):
        standard_deviation = torch.exp(0.5 * logvariance)
        epsilon = torch.randn_like(standard_deviation)
        return mean + epsilon * standard_deviation


    def forward(self, x):
        mean, logvariance = self.encode(x)
        epsilon = torch.randn_like(logvariance)
        z_reparametrize = mean + epsilon * torch.exp(0.5 * logvariance)
        x_reconstructed = self.decode(z_reparametrize)
        return x_reconstructed, mean, logvariance

# if __name__ == "__main__":
#     x = torch.randn(4, 420 * 420)  # Assuming input is a 2D matrix of size (420, 420)
#     vae = Variational2DAutoEncoder(input_dimensions=420 * 420, hidden_dimensions=200, latent_space_dimensions=20)
#     x_reconstructed, mean, logvariance = vae(x)
#     print(x_reconstructed.shape)
#     print(mean.shape)
#     print(logvariance.shape)