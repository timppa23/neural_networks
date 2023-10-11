# %%

import torch
import torchvision.datasets as datasets
from tqdm import tqdm
from torch import nn, optim
from model import VariationalAutoEncoder
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIMENSIONS = 28*28
HIDDEN_DIMENSIONS = 200
LATENT_SPACE_DIMENSIONS = 20
NUMBER_OF_EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 1e-4

dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
training_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
model = VariationalAutoEncoder(INPUT_DIMENSIONS, HIDDEN_DIMENSIONS, LATENT_SPACE_DIMENSIONS).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_function = nn.BCELoss(reduction="sum")


# %%
# Start training
for epoch in range(NUMBER_OF_EPOCHS):
    loop = tqdm(enumerate(training_loader))
    for i, (x, _) in loop:
        # Forward pass
        x = x.to(DEVICE).view(x.shape[0], INPUT_DIMENSIONS)
        x_reconstructed, mu, sigma = model(x)

        # Loss calculation
        reconstruction_loss = loss_function(x_reconstructed, x)
        KL_divergence = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))


        # Backpropagation
        loss = reconstruction_loss + KL_divergence
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())

model = model.to("cpu")

# %%


def inference(digit, num_examples=1):
    images = []
    index = 0
    for x, y in dataset:
        if y == index:
            images.append(x)
            index += 1
        if index == 10:
            break

    encodings_digit = []
    for d in range(10):
        with torch.no_grad():
            mu, sigma = model.encode(images[d].view(1, INPUT_DIMENSIONS))
            print(f"mu: {mu}, sigma: {sigma}")
        encodings_digit.append((mu, sigma))

    mu, sigma = encodings_digit[digit]
    for example in range(num_examples):
        epsilon = torch.randn_like(sigma)
        z = mu + epsilon * sigma
        out = model.decode(z)
        out = out.view(-1, 1, 28, 28)
        save_image(out, f"output/{digit}_{example}.png")

for index in range(10):
    inference(index, num_examples=5)


# %%
