# %%
import os
import sys
import torch.nn.functional as F
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from convo_3d_vae import Convo3DVAE
# This assumes your notebook is in the 3d_audio_convolution directory and you want to import from preprocess
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..', 'preprocess')))
from preprocess import read_file, resample, segment_audio_return, write_example_file, calculate_grouped_value_count
import numpy as np


# Preprocess parameters
number_of_training_songs = 201
number_of_validation_songs = 90
number_of_testing_songs = 10
segment_length_secs = 10
sample_rate = 44100
lower_sample_rate = sample_rate // 10
segment_to_song_coefficient = 1 #int(120 / segment_length_secs)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIMENSION = int(np.sqrt(segment_length_secs * lower_sample_rate))


resampled_songs = []
song_segments = []


for song_id in range(number_of_training_songs + number_of_validation_songs + number_of_testing_songs):
    song = read_file(song_id)
    song = resample(song[0], song[1], lower_sample_rate)
    resampled_songs.append(song)



for song in resampled_songs:
    segments = segment_audio_return(song[0], song[1], segment_length_secs, INPUT_DIMENSION, 5)
    song_segments.append(np.array(segments)[:10])  # Append individual segments

print(f"song_segments count: {len(song_segments)}")

# Split data into training, validation, and testing sets
train_data = song_segments[:number_of_training_songs * segment_to_song_coefficient]
validation_data = song_segments[number_of_training_songs * segment_to_song_coefficient:(number_of_training_songs + number_of_validation_songs) * segment_to_song_coefficient]
test_data = song_segments[(number_of_training_songs + number_of_validation_songs) * segment_to_song_coefficient:]

resampled_songs = []
# song_segments = []

# %% Create DataLoader for training

# Hyperparameters
BATCH_SIZE = 32
NUM_OF_FRAMES = train_data[0].shape[0]
HIDDEN_DIMENSIONS1 = 8400
HIDDEN_DIMENSIONS2 = 4400
HIDDEN_DIMENSIONS3 = 2800
LATENT_SPACE_DIMENSIONS = 840
LEARNING_RATE = 1e-5
NUMBER_OF_EPOCHS = 500
BETA_VALUE = 1
ALPHA_VALUE = 1

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)
validation_loader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

model = Convo3DVAE(INPUT_DIMENSION, NUM_OF_FRAMES, HIDDEN_DIMENSIONS1, HIDDEN_DIMENSIONS2, HIDDEN_DIMENSIONS3, LATENT_SPACE_DIMENSIONS).to(DEVICE)

# Convert the model's parameters to float16
# model = model.half()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_function = nn.MSELoss()

 #%%
# files paths
checkpoint_path = '3dvae_checkpointv2.pth'
final_path = '3dvaev2.pth'
# Check if a checkpoint file exists
if os.path.exists(checkpoint_path):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_loss = checkpoint['best_loss']
    epoch = checkpoint['epoch']
    print(f"Loaded checkpoint. Resuming from epoch {epoch}, best loss: {best_loss}")
    NUMBER_OF_EPOCHS = NUMBER_OF_EPOCHS - epoch

else:
    best_loss = float('inf')


# %% Start training
print(f"Starting training")
for epoch in range(NUMBER_OF_EPOCHS):
    model.train()  # Set model to training mode   
    total_train_loss = 0.0

    loop = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, x in loop:
        # Forward pass
        x = x.clone().detach().to(DEVICE).view(x.shape[0], 1, NUM_OF_FRAMES, INPUT_DIMENSION, INPUT_DIMENSION)

        x_reconstructed, mu, logvar = model(x)

        # Loss calculation
        reconstruction_loss = loss_function(x_reconstructed, x)
        #KL_divergence = -0.5 * torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
        # KL Divergence loss
        KL_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Backpropagation
        loss = (ALPHA_VALUE * reconstruction_loss) + (BETA_VALUE * KL_divergence)


        optimizer.zero_grad()
        loss.backward()
        #grad_norm = get_gradient_norm(model)
        #print(f"Gradient Norm: {grad_norm}")
        optimizer.step()
        
        total_train_loss += loss.item()
        loop.set_description(f"Epoch train [{epoch+1}/{NUMBER_OF_EPOCHS}]")
        loop.set_postfix(loss=total_train_loss / (i + 1), KL_divergence={(BETA_VALUE * KL_divergence.item())}, reconstruction_loss= {(ALPHA_VALUE * reconstruction_loss.item())})


    if epoch % 10 == 0:
        # print(f"x values: {x[0]}")
        # print(f"x_reconstructed values: {x_reconstructed[0]}")
        print(f"mu values: {calculate_grouped_value_count(mu[0])}")
        print(f"sigma values: {calculate_grouped_value_count(logvar[0])}")
        print(f"grouped values x: {calculate_grouped_value_count(x[0])}")
        print(f"grouped values x_reconstructed: {calculate_grouped_value_count(x_reconstructed[0])}")
        torch.save({
            'epoch': 0,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': total_train_loss / len(train_loader),
        }, final_path)
        

    model.eval()
    total_val_loss = 0.0

    with torch.no_grad():
        val_loop = tqdm(enumerate(validation_loader), total=len(validation_loader))
        for i, x in val_loop:
            # Forward pass
            x = x.clone().detach().to(DEVICE).view(x.shape[0], 1, NUM_OF_FRAMES, INPUT_DIMENSION, INPUT_DIMENSION)
            x_reconstructed, mu, logvar = model(x)

            # Loss calculation
            reconstruction_loss = loss_function(x_reconstructed, x)
            #KL_divergence = -0.5 * torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
            # KL Divergence loss
            KL_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = (ALPHA_VALUE * reconstruction_loss) + (BETA_VALUE * KL_divergence)

            total_val_loss += loss.item()
            val_loop.set_description(f"Epoch validation [{epoch+1}/{NUMBER_OF_EPOCHS}]")
            val_loop.set_postfix(loss=total_val_loss / (i + 1), KL_divergence= {(BETA_VALUE * KL_divergence.item())}, reconstruction_loss= {(ALPHA_VALUE * reconstruction_loss.item())})
    
        avg_val_loss = total_val_loss / len(validation_loader)
        # Save checkpoint if the current loss is better than the best loss so far
        if avg_val_loss < best_loss:
            print(f"Saving checkpoint. Loss: {avg_val_loss}")
            best_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
            }, checkpoint_path)



        



# %%
model.eval()
test_loader = DataLoader(test_data[0:1], batch_size=BATCH_SIZE, shuffle=False)

with torch.no_grad():
    val_loop = tqdm(enumerate(test_loader), total=len(test_loader))
    for i, x in val_loop:
        # Forward pass
        x = x.clone().detach().to(DEVICE).view(x.shape[0], 1, NUM_OF_FRAMES, INPUT_DIMENSION, INPUT_DIMENSION)
        x_reconstructed, mu, logvar = model.forward(x)


reconstruction_loss = loss_function(x_reconstructed, x)
print(f"reconstruction loss: {( ALPHA_VALUE * reconstruction_loss)}")
#KL_divergence = -0.5 * torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
# KL Divergence loss
KL_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
print(f"KL divergence: {(BETA_VALUE * KL_divergence)}")
loss = ( ALPHA_VALUE * reconstruction_loss) + (BETA_VALUE * KL_divergence)
print(f"loss: {loss}")

print(f"mu values: {calculate_grouped_value_count(mu[0])}")
print(f"sigma values: {calculate_grouped_value_count(logvar[0])}")
print(f"grouped values x: {calculate_grouped_value_count(x[0])}")
print(f"grouped values x_reconstructed: {calculate_grouped_value_count(x_reconstructed[0])}")


#%%
# Convert the concatenated reconstructed audio to the waveform
reconstructed_waveform = x_reconstructed.cpu().numpy().reshape(-1)
validation_waveform =  x.cpu().numpy().reshape(-1)

# Save the reconstructed audio as a .wav file
write_example_file('3', reconstructed_waveform, validation_waveform, lower_sample_rate)

 #  %%
 