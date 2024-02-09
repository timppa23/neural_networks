# %%
import os
import torch.nn.functional as F
# Change the current working directory
#new_directory = "/Users/michael/Uni/neural_networks/audio_convolution"
#os.chdir(new_directory)
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
print(f"current working directory: {os.getcwd()}")
from variational_autoencoder import Variational2DAutoEncoder 
# Define preprocess functions
import numpy as np
import torchaudio
from torchaudio.transforms import Resample
import torch.autograd.profiler as profiler
from scipy.fftpack import fft, rfft, irfft
import soundfile as sf


# Training parameters
number_of_training_songs = 261
number_of_validation_songs = 30
number_of_testing_songs = 10
segment_length_secs = 3
sample_rate = 44100
lower_sample_rate = sample_rate // 10
segment_to_song_coefficient = int(120 / segment_length_secs)
checkpoint_path = 'vae_checkpoint_dft_8400_3440_2800_sigmoid.pth'
final_path = 'vae_final_dft_8400_3440_2800_sigmoid.pth'


DATA_FILES_WAV = '../autoencoder/raw_audio'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#INPUT_DIMENSION = int(np.sqrt(segment_length_secs * lower_sample_rate))


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
        resample = Resample(orig_freq=sample_rate, new_freq=lower_sample_rate)
        audio_data = resample(audio_data)
        return (audio_data, lower_sample_rate)

def segment_audio_return(audio_data, sample_rate, segment_length_secs):
    # Convert stereo to mono if needed
    if len(audio_data.shape) > 1 and audio_data.shape[0] == 2:
        audio_data = audio_data.mean(axis=0)

    # Calculate the number of samples in the segment
    segment_length_samples = int(sample_rate * segment_length_secs)

    # Create segments
    num_segments = len(audio_data) // segment_length_samples
    segments = []

    for i in range(num_segments):
        start = i * segment_length_samples
        end = start + segment_length_samples
        segment = audio_data[start:end]

        # Reshape the segment into a 2D matrix 
        #reshaped_segment = np.reshape(rfft(segment.numpy()), (INPUT_DIMENSION, INPUT_DIMENSION))
        reshaped_segment = normalize_data_16bit(rfft(segment.numpy()))
        # reshaped_segment = rfft(segment.numpy())

        segments.append(reshaped_segment)

    return segments


def normalize_data_16bit(data):
    # 16-bit range
    min_val = -3394.1180
    max_val = 3319.0127
    
    # Normalize data to be between 0 and 1
    normalized_data = (data - min_val) / (max_val - min_val)

    return normalized_data


def reverse_normalization_16bit(normalized_data):
    # 16-bit range
    min_val = -3394.1180
    max_val = 3319.0127
    
    # Reverse normalization
    data = normalized_data * (max_val - min_val) + min_val
    
    return data

def get_max_value(song_segments):
    max_value = None  # Initialize max_value as None to handle the case where resampled_songs might be empty

    for i, song in enumerate(song_segments):
        # Assuming each 'song' is a list or an array and has at least three elements
        curr_max = max(song)
        print(i)
        if max_value is None or curr_max > max_value:
            max_value = curr_max
            print(max_value)

    return(max_value)

def get_min_value(song_segments):
    min_value = None  # Initialize max_value as None to handle the case where resampled_songs might be empty

    for i, song in enumerate(song_segments):
        # Assuming each 'song' is a list or an array and has at least three elements
        curr_min = min(song)
        print(i)
        if min_value is None or curr_min < min_value:
            min_value = curr_min
            print(min_value)
            

    return(min_value)


def normalize_song_segments():
    # Find the minimum and maximum values in the list
    min_value = get_min_value(song_segments)
    max_value = get_max_value(song_segments)

    # Normalize the list to be between 0 and 1
    normalized_data = [(x - min_value) / (max_value - min_value) for x in song_segments]
    return normalized_data

songs = []
resampled_songs = []

for song_id in range(number_of_training_songs + number_of_validation_songs + number_of_testing_songs):
    print(f"Reading song {song_id}")
    song = read_file(song_id)
    # songs.append(song)
    song = resample(song[0], song[1])
    resampled_songs.append(song)




song_segments = []

for song in resampled_songs:
    segments = segment_audio_return(song[0], song[1], segment_length_secs)
    for segment in segments:
        song_segments.append(segment)  # Append individual segments

print(f"song_segments count: {len(song_segments)}")




# Split data into training, validation, and testing sets
train_data = song_segments[:number_of_training_songs * segment_to_song_coefficient]
validation_data = song_segments[
                  number_of_training_songs * segment_to_song_coefficient:(number_of_training_songs + number_of_validation_songs) * segment_to_song_coefficient]
test_data = song_segments[
            (number_of_training_songs + number_of_validation_songs) * segment_to_song_coefficient:]




#%%

songs = []
#resampled_songs = []
#song_segments = []

checkpoint_path = 'vae_checkpoint_dft_8400_5800_3400_2800_v5.pth'
final_path = 'vae_final_dft_8400_5800_3400_2800_v5.pth'

# %% Create DataLoader for training
BATCH_SIZE = 256
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)
validation_loader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# Initialize the model, optimizer, and loss function
INPUT_DIMENSION = len(train_data[0])
print(f"Input dimension: {INPUT_DIMENSION}")
INPUT_DIMENSIONS = INPUT_DIMENSION
print(f"New input dimension: {INPUT_DIMENSIONS}")
HIDDEN_DIMENSIONS1 = 8400
HIDDEN_DIMENSIONS2 = 5800
HIDDEN_DIMENSIONS3 = 3400
LATENT_SPACE_DIMENSIONS = 2800
LEARNING_RATE = 1e-5
NUMBER_OF_EPOCHS = 500
BETA_VALUE = 1
ALPHA_VALUE = 1e9

model = Variational2DAutoEncoder(INPUT_DIMENSIONS, HIDDEN_DIMENSIONS1, HIDDEN_DIMENSIONS2, HIDDEN_DIMENSIONS3, LATENT_SPACE_DIMENSIONS).to(DEVICE)
# model = Variational2DAutoEncoder(INPUT_DIMENSIONS, HIDDEN_DIMENSIONS1, HIDDEN_DIMENSIONS3, LATENT_SPACE_DIMENSIONS).to(DEVICE)

# Convert the model's parameters to float16
#model = model.half()
data_type = torch.float32
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_function = nn.MSELoss()

#%%
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



def get_gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

 #%%
NUMBER_OF_EPOCHS = 500 
BETA_VALUE = 1
ALPHA_VALUE = 1e10
best_loss = float('inf')

# %% Start training
print(f"Starting training")
for epoch in range(NUMBER_OF_EPOCHS):
    model.train()  # Set model to training mode   
    total_train_loss = 0.0

    loop = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, x in loop:
        # Forward pass
        # x = torch.tensor(x, dtype=data_type).to(DEVICE).view(x.shape[0], 1, INPUT_DIMENSION, INPUT_DIMENSION)
        # x = torch.tensor(x, dtype=data_type).to(DEVICE).view(x.shape[0], 1, INPUT_DIMENSION)
        x = x.clone().detach().to(DEVICE).view(x.shape[0], 1, INPUT_DIMENSION)


        x_reconstructed, mu, sigma = model(x)

        # Loss calculation
        #x = x.view(x.size(0), -1)
        reconstruction_loss = loss_function(x_reconstructed, x)
        # KL_divergence = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
        KL_divergence = -0.5 * torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
        # Backpropagation
        loss = (ALPHA_VALUE * reconstruction_loss) + (BETA_VALUE * KL_divergence)
        # loss = reconstruction_loss +  KL_divergence
        optimizer.zero_grad()
        loss.backward()
        #grad_norm = get_gradient_norm(model)
        #print(f"Gradient Norm: {grad_norm}")
        optimizer.step()
        
        total_train_loss += loss.item() / x.size(0)
        loop.set_description(f"Epoch train [{epoch+1}/{NUMBER_OF_EPOCHS}]")
        loop.set_postfix(loss=total_train_loss / (i + 1), KL_divergence={(BETA_VALUE * KL_divergence.item()) / x.size(0)}, reconstruction_loss= {(ALPHA_VALUE * reconstruction_loss.item()) / x.size(0)})

    #print(f"Epoch [{epoch + 1}/{NUMBER_OF_EPOCHS}], AVG Training Loss: {total_train_loss / len(train_loader)}, KL divergence: {(BETA_VALUE * KL_divergence) / x.size(0)}, reconstruction_loss: {(ALPHA_VALUE * reconstruction_loss) / x.size(0)}")

    model.eval()
    total_val_loss = 0.0   

    with torch.no_grad():
        val_loop = tqdm(enumerate(validation_loader), total=len(validation_loader))
        for i, x in val_loop:
            # Forward pass
            #x = torch.tensor(x, dtype=data_type).to(DEVICE).view(x.shape[0], 1, INPUT_DIMENSION, INPUT_DIMENSION)
            x = x.clone().detach().to(DEVICE).view(x.shape[0], 1, INPUT_DIMENSION)
            x_reconstructed, mu, sigma = model(x)

            # Loss calculation
            #x = x.view(x.size(0), -1)
            reconstruction_loss = loss_function(x_reconstructed, x)
            #KL_divergence = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
            KL_divergence = -0.5 * torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
            loss = (ALPHA_VALUE * reconstruction_loss) + (BETA_VALUE * KL_divergence)

            total_val_loss += loss.item() / x.size(0)
            val_loop.set_description(f"Epoch validation [{epoch+1}/{NUMBER_OF_EPOCHS}]")
            val_loop.set_postfix(loss=total_val_loss / (i + 1), KL_divergence= {(BETA_VALUE * KL_divergence.item())   / x.size(0)}, reconstruction_loss= {(ALPHA_VALUE * reconstruction_loss.item()) / x.size(0)})
    
        avg_val_loss = total_val_loss / len(validation_loader)
        #print(f"Epoch [{epoch + 1}/{NUMBER_OF_EPOCHS}], AVG Validation Loss: {avg_val_loss}, KL divergence: {(BETA_VALUE * KL_divergence)   / x.size(0)}, reconstruction_loss: {(ALPHA_VALUE * reconstruction_loss) / x.size(0)}")
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
  


 #%%
            
final_path = 'vae_final_dft_8400_5800_3400_2800_v4.pth'            
torch.save({
    'epoch': 0,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'best_loss': avg_val_loss,
}, final_path)

    
   # %%
# Take 32 segments from the validation set
x_validation_segments = test_data[:32]  # Assuming validation_data is a list of segments

# Convert each segment to a tensor and ensure its shape matches INPUT_DIMENSION
# This assumes that each segment is already in the shape (INPUT_DIMENSION, INPUT_DIMENSION)
for i in range(len(x_validation_segments)):
    x_validation_segments[i] = torch.tensor(x_validation_segments[i], dtype=data_type).to(DEVICE)

# Stack the segments along a new dimension to form a batch
x_validation = torch.stack(x_validation_segments)

# Ensure the shape is (32, 1, INPUT_DIMENSION, INPUT_DIMENSION)
x_validation = x_validation.view(32, 1, INPUT_DIMENSION, INPUT_DIMENSION)


# %% Forward pass through the model

import soundfile as sf
model.eval()
with torch.no_grad():
    mean, logvariance = model.encode(x_validation)
    z_reparametrize = model.reparameterize(mean, logvariance)
    x_reconstructed = model.decode(z_reparametrize)
    x_reconstructed = x_reconstructed.view(x_reconstructed.size(0), -1)

# Convert reconstructed tensor segments to audio waveform
reconstructed_audio = x_reconstructed.cpu().numpy()

# Here, you need to reshape and concatenate the segments to get a continuous audio stream
# You can use numpy's concatenate function for this if needed

# Convert the concatenated reconstructed audio to the waveform
reconstructed_waveform = reconstructed_audio.reshape(-1)
validation_waveform = x_validation.cpu().numpy().reshape(-1)

# Save the reconstructed audio as a .wav file
sf.write("reconstructed_audio3.wav", reconstructed_waveform, sample_rate)
sf.write("x_validation_segments3.wav", validation_waveform, sample_rate)

# (Optional) If you want to play the reconstructed audio, you can use torchaudio
# torchaudio.play(torch.tensor(reconstructed_waveform), sample_rate)

# %%
model.eval()
test_loader = DataLoader(train_data[96:352], batch_size=BATCH_SIZE, shuffle=False)

with torch.no_grad():
    val_loop = tqdm(enumerate(test_loader), total=len(test_loader))
    for i, x in val_loop:
        # Forward pass
        # x = torch.tensor(x, dtype=data_type).to(DEVICE).view(x.shape[0], 1, INPUT_DIMENSION, INPUT_DIMENSION)
        x = torch.tensor(x, dtype=data_type).to(DEVICE).view(x.shape[0], 1, INPUT_DIMENSION)
        x_reconstructed, mu, sigma = model.forward(x)

#%%
reconstruction_loss = loss_function(x_reconstructed, x)
print(f"reconstruction loss: {( ALPHA_VALUE * reconstruction_loss) / x.size(0)}")
-torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
print(f"KL divergence: {(BETA_VALUE * KL_divergence) / x.size(0)}")
loss = ( ALPHA_VALUE * reconstruction_loss) + (BETA_VALUE * KL_divergence)
print(f"loss: {loss / x.size(0)}")

#%%

reconstructed_audio_reversed = reverse_normalization_16bit(x_reconstructed.cpu().numpy())
validation_reversed = reverse_normalization_16bit(x.cpu().numpy())


#%%
# Convert the concatenated reconstructed audio to the waveform
reconstructed_waveform = irfft(reconstructed_audio_reversed).reshape(-1)
validation_waveform = irfft(validation_reversed).reshape(-1)
#%%  

validation_waveform.shape

reconstruction_loss = loss_function(x_reconstructed, x)
print(f"reconstruction loss: {reconstruction_loss}")
-torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
print(f"KL divergence: {KL_divergence}")
loss = reconstruction_loss + (BETA_VALUE * KL_divergence)
print(f"loss: {loss}")

#%%

# Save the reconstructed audio as a .wav file
# reconstruction loss: 589.0449829101562
# KL divergence: 0.0060395002365112305
sf.write("reconstructed_audio11.wav", reconstructed_waveform, (sample_rate // 10))
sf.write("x_validation_segments11.wav", validation_waveform, (sample_rate // 10) )

 #  %%
 