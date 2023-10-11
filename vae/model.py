import torch
import torch.nn.functional as F
from torch import nn


# Input image -> Hidden dimension -> mean, standard deviation -> Parametrzaion trick -> latent space -> Decoder -> Output image
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dimensions, hidden_dimesions = 200, latent_space_dimensions = 20):
        super().__init__()
        # encoder
        self.image_to_hidden_dimension = nn.Linear(input_dimensions, hidden_dimesions)
        self.hidden_to_mu = nn.Linear(hidden_dimesions, latent_space_dimensions)
        self.hidden_to_sigma = nn.Linear(hidden_dimesions, latent_space_dimensions)

        # decoder
        self.z_to_hidden_dimension = nn.Linear(latent_space_dimensions, hidden_dimesions)
        self.hidden_to_image = nn.Linear(hidden_dimesions, input_dimensions)

        self.relu = nn.ReLU()


    def encode(self, x):
        hidden = self.relu(self.image_to_hidden_dimension(x))
        mu, sigma = self.hidden_to_mu(hidden), self.hidden_to_sigma(hidden)

        return mu, sigma
    

    def decode(self, z):
        hidden = self.relu(self.z_to_hidden_dimension(z))
        return torch.sigmoid(self.hidden_to_image(hidden))
        

    def forward(self, x):
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)
        z_reparametrize = mu + epsilon * sigma
        x_reconstructed = self.decode(z_reparametrize)

        return x_reconstructed, mu, sigma


if __name__ == "__main__":
    x = torch.randn(4, 28*28) # 28*28 = 784
    vae = VariationalAutoEncoder(input_dimensions=28*28, hidden_dimesions=200, latent_space_dimensions=20)
    x_reconstructed, mu, sigma = vae(x)
    print(x_reconstructed.shape)
    print(mu.shape)
    print(sigma.shape)