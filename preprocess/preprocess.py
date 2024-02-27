import os
import numpy as np
import torchaudio
from torchaudio.transforms import Resample
# from scipy.fftpack import fft, rfft, irfft
import soundfile as sf
from collections import Counter

DATA_FILES_WAV = '../autoencoder/raw_audio'

def normalize_data(data, min_val, max_val):
    # Normalize data to be between 0 and 1
    normalized_data = (data - min_val) / (max_val - min_val)

    return normalized_data


def reverse_normalization(normalized_data, min_val, max_val):    
    # Reverse normalization
    data = normalized_data * (max_val - min_val) + min_val
    
    return data

def get_max_value(song_segments):
    max_value = None  # Initialize max_value as None to handle the case where resampled_songs might be empty

    for i, song in enumerate(song_segments):
        # Assuming each 'song' is a list or an array and has at least three elements
        curr_max = max(song)
        if max_value is None or curr_max > max_value:
            max_value = curr_max

    return(max_value)

def get_min_value(song_segments):
    min_value = None  # Initialize max_value as None to handle the case where resampled_songs might be empty

    for i, song in enumerate(song_segments):
        # Assuming each 'song' is a list or an array and has at least three elements
        curr_min = min(song)
        if min_value is None or curr_min < min_value:
            min_value = curr_min
            
    return(min_value)


def normalize_song_segments(song_segments):
    # Find the minimum and maximum values in the list
    min_value = get_min_value(song_segments)
    max_value = get_max_value(song_segments)

    
    return normalize_data(song_segments, min_value, max_value)


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


def resample(audio_data, sample_rate, lower_sample_rate):
        # Decrease the sample rate by a factor of ten
        resample = Resample(orig_freq=sample_rate, new_freq=lower_sample_rate)
        audio_data = resample(audio_data)
        return (audio_data, lower_sample_rate)


def segment_audio_return(audio_data, sample_rate, segment_length_secs, input_dimension, round_to=5):
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
        # segment = [round(float(num), round_to) for num in segment.numpy()]

        # Reshape the segment into a 2D matrix 
        reshaped_segment = np.reshape(segment.numpy(), (input_dimension, input_dimension))
        segments.append(reshaped_segment)

    return segments


def get_gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def write_example_file(number, reconstructed_waveform, validation_waveform, sample_rate):
    sf.write(f"reconstructed_audio_{number}.wav", reconstructed_waveform, sample_rate)
    sf.write(f"validation_audio_{number}.wav", validation_waveform, sample_rate)


def calculate_grouped_value_count(x, decimal_places=7):
    # Convert the tensor to a NumPy array
    array = x.view(-1).cpu().detach().numpy()
    
    # Round the numbers to the specified number of decimal places
    # This also avoids issues with scientific notation
    rounded = [round(float(num), decimal_places) for num in array]
    
    # Convert rounded numbers to strings for consistent handling
    grouped = [f"{num:.{decimal_places}f}" for num in rounded]

    # Count occurrences of each rounded value
    counts = Counter(grouped)

    # Get the top 10 most common
    top_ten = counts.most_common(10)

    return top_ten