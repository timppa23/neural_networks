# %% imports
import torch
import os
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torchaudio
from torchaudio.transforms import Resample
import torch.autograd.profiler as profiler
from convo_autoencoder import ConvoAutoencoder
import soundfile as sf

from scipy.fftpack import fft, rfft, irfft


# Training parameters
number_of_training_songs = 261
number_of_validation_songs = 30
number_of_testing_songs = 10
segment_length_secs = 10
sample_rate = 44100
lower_sample_rate = sample_rate // 10
segment_to_song_coefficient = int(120 / segment_length_secs)
checkpoint_path = 'convo_autoencoder_dft.pth'
final_path = 'convo_autoencoder_final_dft.pth'


DATA_FILES_WAV = '../autoencoder/raw_audio'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIMENSION = int(np.sqrt(segment_length_secs * lower_sample_rate))


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


def resample(audio_data, sample_rate):
            # Decrease the sample rate by a factor of ten
        resample = Resample(orig_freq=sample_rate, new_freq=(sample_rate // 10))
        audio_data = resample(audio_data)
        return (audio_data, (sample_rate // 10))

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

        # segments.append(rfft(segment.numpy()))
        # segments.append(segment.numpy())
        segments.append(np.reshape(rfft(segment.numpy()), (INPUT_DIMENSION, INPUT_DIMENSION)))

    print(f"returning count: {segments[0].shape}")
    print(f"returning count: {len(segments)}")
    return segments



songs = []
resampled_songs = []

for song_id in range(number_of_training_songs + number_of_validation_songs + number_of_testing_songs):
    song_id = song_id 
    print(f"Reading song {song_id}")
    song = read_file(song_id)
    # songs.append(song)
    song = resample(song[0], song[1])
    resampled_songs.append(song)

print(f"Song count: {len(songs)}")
# #%%
# resampled_songs[0][0].mean(axis=0).shape



# #%%
# import soundfile as sf
# sf.write("resampled_audio2.wav", resampled_songs[85][0].mean(axis=0), resampled_songs[85][1])


# # %%
song_segments = []
print(f"Song count: {len(songs)}")

for song in resampled_songs:
    segments = segment_audio_return(song[0], song[1], segment_length_secs)
    print(f"Adding segments: {len(segments)}")
    print(f"Segments total count: {len(song_segments)}")
    for segment in segments:
        song_segments.append(segment)  # Append individual segments

print(f"song_segments count: {len(song_segments)}")

# Split data into training, validation, and testing sets
train_data = song_segments[:(number_of_training_songs * segment_to_song_coefficient)]
validation_data = song_segments[
                  number_of_training_songs * segment_to_song_coefficient:(number_of_training_songs + number_of_validation_songs) * segment_to_song_coefficient]
test_data = song_segments[
            (number_of_training_songs + number_of_validation_songs) * segment_to_song_coefficient:]

# %%

train_data[0].shape
songs = []
# resampled_songs = []
# song_segments = []

# %%
BATCH_SIZE = 32
print(f"train_loader")
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# Initialize the model, optimizer, and loss function
print(f"Input dimension: {INPUT_DIMENSION}")    
LEARNING_RATE = 1e-5
NUMBER_OF_EPOCHS = 500

print(f"model")
model = ConvoAutoencoder(INPUT_DIMENSION, first_layer_hidden_dimensions=8400, second_layer_hidden_dimensions=3440, latent_space_dimensions=2800).to(DEVICE)
# Convert the model's parameters to float16
#model = model.half()
data_type = torch.float32
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_function = nn.MSELoss()

# Checkpoint file path

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
    NUMBER_OF_EPOCHS = 50# NUMBER_OF_EPOCHS - epoch

else:
    best_loss = float('inf')

# %%
NUMBER_OF_EPOCHS = 500
epoch = 0           

#%%
print(f"Starting training")
# Profile memory usage
# with profiler.profile(use_cuda=True) as prof:
for epoch in range(NUMBER_OF_EPOCHS):
    model.train()  # Set model to training mode
    total_train_loss = 0.0

    loop = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, x in loop:
        # Forward pass
        #print(f"Forward pass: {i}")
        x = x.clone().detach().to(DEVICE).view(x.shape[0], 1, INPUT_DIMENSION, INPUT_DIMENSION)

        x_reconstructed = model(x)

        # Loss calculation
        reconstruction_loss = loss_function(x_reconstructed, x)

        # Backpropagation
        loss = reconstruction_loss
        # loss /= accumulation_steps  # Normalize loss for gradient accumulation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item() / x.size(0)
        loop.set_description(f"Epoch train [{epoch+1}/{NUMBER_OF_EPOCHS}]")
        loop.set_postfix(loss=total_train_loss / (i + 1))


    avg_loss = total_train_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{NUMBER_OF_EPOCHS}], AVG Training Loss: {total_train_loss / (i + 1):.4f}")

    

    model.eval()
    total_val_loss = 0.0   

    with torch.no_grad():
        val_loop = tqdm(enumerate(validation_loader), total=len(validation_loader))
        for i, x in val_loop:
            # Forward pass
            x = x.clone().detach().to(DEVICE).view(x.shape[0], 1, INPUT_DIMENSION, INPUT_DIMENSION)
            x_reconstructed = model(x)

            # Loss calculation
            reconstruction_loss = loss_function(x_reconstructed, x)
            loss = reconstruction_loss

            total_val_loss += loss.item() / x.size(0)
            val_loop.set_description(f"Epoch validation [{epoch+1}/{NUMBER_OF_EPOCHS}]")
            val_loop.set_postfix(loss=total_val_loss / (i + 1))
    
        avg_val_loss = total_val_loss / len(validation_loader)
        print(f"Epoch [{epoch + 1}/{NUMBER_OF_EPOCHS}], AVG Validation Loss: {avg_val_loss:.4f}")
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
total_val_loss = 0.0   
test_loader = DataLoader(test_data[64:96], batch_size=BATCH_SIZE, shuffle=False)

with torch.no_grad():
    val_loop = tqdm(enumerate(test_loader), total=len(test_loader))
    for i, x in val_loop:
        # Forward pass
        x = x.clone().detach().to(DEVICE).view(x.shape[0], 1, INPUT_DIMENSION, INPUT_DIMENSION)
        z = model.encode(x)
        x_reconstructed = model.decode(z)


# Here, you need to reshape and concatenate the segments to get a continuous audio stream
# You can use numpy's concatenate function for this if needed

#%%
# Convert the concatenated reconstructed audio to the waveform
reconstructed_waveform = irfft(x_reconstructed.view(x_reconstructed.size(0), -1).cpu().numpy()).reshape(-1)
validation_waveform = irfft(x.view(x_reconstructed.size(0), -1).cpu().numpy()).reshape(-1)
 

reconstruction_loss = loss_function(x_reconstructed.view(x_reconstructed.size(0), -1).reshape(-1), x.view(x.size(0), -1).reshape(-1))
print(f"reconstruction loss: {reconstruction_loss.item() / x.size(0)}")

reconstruction_loss = loss_function(x_reconstructed, x)
print(f"reconstruction loss: {reconstruction_loss.item()}")


#%%

# Save the reconstructed audio as a .wav file
sf.write("reconstructed_audio5.wav", reconstructed_waveform, (sample_rate // 10))
sf.write("x_validation_segments5.wav", validation_waveform, (sample_rate // 10) )
 # %%