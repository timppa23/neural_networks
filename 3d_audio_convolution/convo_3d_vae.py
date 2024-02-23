import torch
import torch.nn as nn
import torch.nn.functional as F

class Convo3DVAE(nn.Module):
    def __init__(self, input_dimension, num_frames=12, hidden_dimensions1=8400, hidden_dimensions2=5400, hidden_dimensions3=3440, latent_space_dimensions=2800):
        super().__init__()

        self.debug = False
        
        middle_channels = 1
        out_channels = 4
        self.out_channels = out_channels
        self.frame_height = input_dimension
        self.frame_width = input_dimension
        self.num_frames = num_frames

        # Define 3D convolution parameters
        conv1_params = [3, 2, 1]  # kernel_size, stride, padding for the first conv layer
        conv2_params = [3, 2, 1]  # kernel_size, stride, padding for the second conv layer

        # Encoder
        # self.convolutional_layer1 = nn.Conv3d(in_channels=1, out_channels=middle_channels, kernel_size=conv1_params[0], stride=conv1_params[1], padding=conv1_params[2])
        self.convolutional_layer2 = nn.Conv3d(in_channels=middle_channels, out_channels=out_channels, kernel_size=conv2_params[0], stride=conv2_params[1], padding=conv2_params[2])

        # Calculate the output dimensions after convolutional layers to adjust the input dimensions for the linear layer
        def conv3d_output_size(size, kernel_size, stride, padding):
            return (size - kernel_size + 2 * padding) // stride + 1

        self.conv_height = conv3d_output_size(input_dimension, conv1_params[0], conv1_params[1], conv1_params[2]) #, conv2_params[0], conv2_params[1], conv2_params[2])
        self.conv_width = self.conv_height  # Assuming square frames
        self.conv_depth = conv3d_output_size(self.num_frames, conv1_params[0], conv1_params[1], conv1_params[2]) # , conv2_params[0], conv2_params[1], conv2_params[2])

        print(f"Convolutional layer output dimensions: {self.conv_depth} x {self.conv_height} x {self.conv_width} x {out_channels}")

        linear_input_dimensions = self.conv_depth * self.conv_height * self.conv_width * out_channels

        #self.hidden_layer_encoder1 = nn.Linear(in_features=linear_input_dimensions, out_features=hidden_dimensions1)
        self.hidden_layer_encoder2 = nn.Linear(in_features=linear_input_dimensions, out_features=hidden_dimensions2)
        self.hidden_layer_encoder3 = nn.Linear(in_features=hidden_dimensions2, out_features=hidden_dimensions3)
        self.mu_layer = nn.Linear(in_features=hidden_dimensions3, out_features=latent_space_dimensions)
        self.sigma_layer = nn.Linear(in_features=hidden_dimensions3, out_features=latent_space_dimensions)

        # Decoder
        self.hidden_layer_decoder = nn.Linear(in_features=latent_space_dimensions, out_features=hidden_dimensions3)
        self.reconstruction_layer1 = nn.Linear(in_features=hidden_dimensions3, out_features=hidden_dimensions2)
        self.reconstruction_layer2 = nn.Linear(in_features=hidden_dimensions2, out_features=linear_input_dimensions)
        #self.reconstruction_layer3 = nn.Linear(in_features=hidden_dimensions1, out_features=linear_input_dimensions)  # Adjusted to match encoder's flattened output

        self.deconvolutional_layer1 = nn.ConvTranspose3d(in_channels=out_channels, out_channels=middle_channels, kernel_size=conv2_params[0], stride=conv2_params[1], padding=conv2_params[2], output_padding=1)
        #self.deconvolutional_layer2 = nn.ConvTranspose3d(in_channels=middle_channels, out_channels=1, kernel_size=conv1_params[0], stride=conv1_params[1], padding=conv1_params[2], output_padding=1)

        self.relu_activation = nn.ReLU()

    def encode(self, x):
        print("x.shape1", x.shape) if self.debug else None
        # x = self.relu_activation(self.convolutional_layer1(x))
        print("x.shape2", x.shape) if self.debug else None
        x = self.relu_activation(self.convolutional_layer2(x))
        print("x.shape3", x.shape) if self.debug else None
        x = x.view(x.size(0), -1)  # Flatten for linear layers
        print("x.shape4", x.shape) if self.debug else None
        # x = self.relu_activation(self.hidden_layer_encoder1(x))
        x = self.relu_activation(self.hidden_layer_encoder2(x))
        x = self.relu_activation(self.hidden_layer_encoder3(x))
        mu = self.mu_layer(x)
        sigma = torch.sqrt(torch.exp(self.sigma_layer(x)))
        return mu, sigma

    def decode(self, z):
        x = self.relu_activation(self.hidden_layer_decoder(z))
        x = self.relu_activation(self.reconstruction_layer1(x))
        x = self.relu_activation(self.reconstruction_layer2(x))
        # x = self.relu_activation(self.reconstruction_layer3(x))  # The output here should match the flattened size before conv layers
        print("x.shape5", x.shape) if self.debug else None 
        x = x.view(x.size(0), self.out_channels, self.conv_depth, self.conv_height, self.conv_width)  # Reshape to match the output of the last convolutional layer dimensions
        print("x.shape6", x.shape) if self.debug else None
        # x = self.relu_activation(self.deconvolutional_layer1(x))
        print("x.shape7", x.shape) if self.debug else None
        x_reconstructed = self.deconvolutional_layer1(x)
        print("x.shape8", x_reconstructed.shape) if self.debug else None
        return x_reconstructed

    def reparameterize(self, mu, sigma):
        epsilon = torch.randn_like(sigma)
        return mu + epsilon * sigma

    def forward(self, x):
        mu, sigma = self.encode(x)
        z_reparametrize = self.reparameterize(mu, sigma)
        x_reconstructed = self.decode(z_reparametrize)
        return x_reconstructed, mu, sigma