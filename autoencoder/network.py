# %% imports
import preprocess_data
import tensorflow as tf
import numpy as np
from autoencoder import Autoencoder
import librosa


# %% 
# Parameters
learning_rate = 0.001
latent_space_dimensions = 1200
number_of_training_songs = 60
number_of_validation_songs = 10
number_of_testing_songs = 2
segment_length_secs = 0.28
segment_to_song_coefficient = int(120/segment_length_secs)
epochs = 50
sample_rate = 44100

# %% 
# Generate data
songs = []
song_segments = []


# %% 
# Get songs
for song_id in range(number_of_training_songs + number_of_validation_songs + number_of_testing_songs):
    print(f"Reading song {song_id}")
    songs.append(preprocess_data.read_file(song_id))

print(f"Song count: {len(songs)}")


# %% 
# Get segments
for song in songs:
    segments = preprocess_data.segment_audio_return_rfft(song[0], song[1], segment_length_secs)
    print(f"Adding segments: {len(segments)}")
    print(f"Segments total count: {len(song_segments)}")
    for segment in segments:
        song_segments.append(segment)  # Append individual segments



# %% 

input_length = len(song_segments[0])

# Convert song segments to a NumPy array
song_segments = np.array(song_segments)

print(f"Segment count: {len(song_segments)}")
print(f"Segment shape: {song_segments.shape}")
print(f"First segment: {song_segments[0]}")
print(f"Input length: {input_length}")
# %% 
# Split data into training, validation, and testing sets
train_data = song_segments[:number_of_training_songs * segment_to_song_coefficient]
validation_data = song_segments[number_of_training_songs * segment_to_song_coefficient:(number_of_training_songs + number_of_validation_songs) * segment_to_song_coefficient]
test_data = song_segments[(number_of_training_songs + number_of_validation_songs) * segment_to_song_coefficient:]


# %%
print(f"training data count: {len(train_data)}")
print(f"traning data shape: {train_data.shape}")
print(f"First song: {train_data[0]}")
print(f"First song shape: {train_data[0].shape}")

# %% 
# Create an instance of the Autoencoder
autoencoder = Autoencoder(input_length=input_length)

# Compile the autoencoder model
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                   loss='mean_squared_error')

# Train the model using the fit() method
autoencoder.fit(train_data, train_data,
                epochs=epochs,
                verbose=1,
                validation_data=(validation_data, validation_data))

# %%
# Save or load model

autoencoder = tf.keras.models.load_model('3_layers_autoencoder_rfft')
# autoencoder.save("3_layers_autoencoder_rfft")


# %%
# Predict on validation data
reconstructed_test = autoencoder.predict(test_data)



# Evaluate the model on test data if needed
test_loss = autoencoder.evaluate(test_data, test_data)
print(f"Test loss: {test_loss}")


# %%
# Save original and reconstructed segments to the output folder
output_folder = 'output4'

original_segment = test_data[0:34]
reconstructed_segment = reconstructed_test[0:34]


# Save original and reconstructed segments as WAV files
preprocess_data.save_to_wav(segment_length_secs, original_segment, reconstructed_segment, sample_rate, output_folder, 0)

#     original_audio = np.array(irfft(original_segment)).flatten()
original_song = test_data[0:(segment_to_song_coefficient - 1)]
reconstructed_song = reconstructed_test[0:(segment_to_song_coefficient - 1)]


# Save original and reconstructed segments as WAV files
preprocess_data.save_to_wav(120, original_song, reconstructed_song, sample_rate, output_folder, 1)










# %%
""" 
# TESTING
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Load an audio file
audio_file = "audio_wav/0.wav"
y, sr = librosa.load(audio_file)

# Compute chroma features
chroma = librosa.feature.chroma_stft(y=y, sr=sr)

# Display the chroma feature as a heatmap
plt.figure(figsize=(10, 4))
librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
plt.colorbar()
plt.title('Chroma Feature')
plt.show()





# %%

import librosa
import librosa.display
import matplotlib.pyplot as plt

# Load an audio file
audio_file = "audio_wav/0.wav"
y, sr = librosa.load(audio_file)

# Compute MFCCs
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # You can choose the number of coefficients (n_mfcc) you want to compute

# Display the MFCCs as a heatmap
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCCs')
plt.show()

 """
""" 
# %%

def segment_audio(audio_data, sample_rate, segment_length_secs):
    # Convert stereo to mono if needed
    if len(audio_data.shape) > 1 and audio_data.shape[1] == 2:
        audio_data = audio_data.mean(axis=1)

    # Calculate the number of samples in the segment
    segment_length_samples = int(sample_rate * segment_length_secs)

    # Create segments
    num_segments = len(audio_data) // segment_length_samples
    segments = []

    for i in range(num_segments):
        start = i * segment_length_samples
        end = start + segment_length_samples
        segment = audio_data[start:end]

        # Calculate Chroma features for the segment using librosa
        chroma = librosa.feature.chroma_stft(y=segment, sr=sample_rate)

        segments.append(chroma)

    return segments

song = songs[0]

segments =  segment_audio(song[0], song[1], segment_length_secs)

for segment in segments:
        song_segments.append(segment)  # Append individual segments

input_length = len(song_segments[0])
# Convert song segments to a NumPy array
song_segments = np.array(song_segments)

print(f"Segment count: {len(song_segments)}")
print(f"Segment shape: {song_segments.shape}")
print(f"First segment: {song_segments[0]}")
print(f"First segment: {song_segments[0].shape}") """