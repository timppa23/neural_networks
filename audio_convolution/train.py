# %%
import os

# Change the current working directory
#new_directory = "/Users/michael/Uni/neural_networks/audio_convolution"
#os.chdir(new_directory)
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from model1 import Variational2DAutoEncoder  # Assuming the new model is saved in 'model.py'
# Define preprocess functions
import numpy as np
import torchaudio
import torch.autograd.profiler as profiler




# Training parameters
number_of_training_songs = 20
number_of_validation_songs = 5
number_of_testing_songs = 5
segment_length_secs = 1
sample_rate = 44100
segment_to_song_coefficient = int(120 / segment_length_secs)

DATA_FILES_WAV = 'raw_audio'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIMENSION = int(np.sqrt(segment_length_secs * sample_rate))


def read_file(file_name):
    try:
        file_path = f"{DATA_FILES_WAV}/{file_name}.wav"

        # Check if the file exists before attempting to load it
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file '{file_name}' does not exist.")

        audio, sample_rate = torchaudio.load(file_path, format='wav')
        return audio, sample_rate
    except Exception as e:
        print(f"Error loading file '{file_name}': {e}")
        return None, None


def segment_audio_return(audio_data, sample_rate, segment_length_secs):
    # Convert stereo to mono if needed
    print(f"audio_data shape: {audio_data.shape}")
    if len(audio_data.shape) > 1 and audio_data.shape[0] == 2:
        audio_data = audio_data.mean(axis=0)

    print(f"audio_data shape: {audio_data.shape}")
    # Calculate the number of samples in the segment
    segment_length_samples = int(sample_rate * segment_length_secs)

    # Create segments
    num_segments = len(audio_data) // segment_length_samples
    segments = []

    for i in range(num_segments):
        start = i * segment_length_samples
        end = start + segment_length_samples
        segment = audio_data[start:end]

        # Reshape the segment into a 2D matrix (420, 420)
        reshaped_segment = np.reshape(segment, (INPUT_DIMENSION, INPUT_DIMENSION))

        segments.append(reshaped_segment)

    print(f"returning count: {len(segments)}")
    return segments



# %% Get songs
songs = []

for song_id in range(number_of_training_songs + number_of_validation_songs + number_of_testing_songs):
    print(f"Reading song {song_id}")
    songs.append(read_file(song_id))

print(f"Song count: {len(songs)}")

# %% Get segments
song_segments = []
print(f"Song count: {len(songs)}")

for song in songs:
    segments = segment_audio_return(song[0], song[1], segment_length_secs)
    print(f"Adding segments: {len(segments)}")
    print(f"Segments total count: {len(song_segments)}")
    for segment in segments:
        song_segments.append(segment)  # Append individual segments

print(f"song_segments count: {len(song_segments)}")

# Split data into training, validation, and testing sets
train_data = song_segments[:number_of_training_songs * segment_to_song_coefficient]
validation_data = song_segments[
                  number_of_training_songs * segment_to_song_coefficient:(number_of_training_songs + number_of_validation_songs) * segment_to_song_coefficient]
test_data = song_segments[
            (number_of_training_songs + number_of_validation_songs) * segment_to_song_coefficient:]

# %% Create DataLoader for training
BATCH_SIZE = 16  # Smaller batch size
print(f"train_loader")
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

# Initialize the model, optimizer, and loss function

INPUT_DIMENSIONS = INPUT_DIMENSION * INPUT_DIMENSION
HIDDEN_DIMENSIONS = 200
LATENT_SPACE_DIMENSIONS = 20
LEARNING_RATE = 1e-4

print(f"model")
model = Variational2DAutoEncoder(INPUT_DIMENSIONS, HIDDEN_DIMENSIONS, LATENT_SPACE_DIMENSIONS).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_function = nn.BCELoss(reduction="sum")

# Checkpoint file path
checkpoint_path = 'vae_convolutional_checkpoint.pth'
print(f"checkpoint_path: {checkpoint_path}")
print(f"checkpoint_path: {os.path.exists(checkpoint_path)}")
# Check if a checkpoint file exists
if os.path.exists(checkpoint_path):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_loss = checkpoint['best_loss']
    epoch = checkpoint['epoch']
    print(f"Loaded checkpoint. Resuming from epoch {epoch}, best loss: {best_loss}")

else:
    best_loss = float('inf')

# %% Start training
NUMBER_OF_EPOCHS = 20
accumulation_steps = 4  # Gradient accumulation over 4 batches
print(f"Starting training")

# Profile memory usage
with profiler.profile(use_cuda=True) as prof:
    for epoch in range(NUMBER_OF_EPOCHS):
        loop = tqdm(enumerate(train_loader))
        for i, x in loop:
            # Forward pass
            print(f"Forward pass: {i}")
            x = torch.tensor(x, dtype=torch.float32).to(DEVICE).view(x.shape[0], 1, 420, 420)
            x_reconstructed, mu, sigma = model(x)

            # Loss calculation
            reconstruction_loss = loss_function(x_reconstructed, x)
            KL_divergence = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
            print(f"KL_divergence: {KL_divergence}")
            print(f"reconstruction_loss: {reconstruction_loss}")

            # Backpropagation
            print(f"Backpropagation: {i}")
            loss = reconstruction_loss + KL_divergence
            loss /= accumulation_steps  # Normalize loss for gradient accumulation
            optimizer.zero_grad()
            loss.backward()

            # Accumulate gradients
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            loop.set_postfix(loss=loss.item())

            # Print memory usage profile
            print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))

    # Save checkpoint if the current loss is better than the best loss so far
    if loss.item() < best_loss:
        print(f"Saving checkpoint. Loss: {loss.item()}")
        best_loss = loss.item()
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss,
        }, checkpoint_path)

# Save the final trained model
torch.save(model.state_dict(), 'vae_convolutional_final.pth')


# %%
