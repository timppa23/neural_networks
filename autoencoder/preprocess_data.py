from pydub import AudioSegment
from glob import iglob
from scipy.fftpack import rfft, irfft
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.io import wavfile  # Import the wavfile module
import librosa

DATA_FILES_MP3 = 'audio'
DATA_FILES_WAV = 'audio_wav'
file_arr = []
current_batch = 0



def read_file(file_name):

    audio_binary = tf.io.read_file(DATA_FILES_WAV + "/"+str(file_name)+".wav")
    wav_decoder = tf.audio.decode_wav(audio_binary, desired_channels=2)
    sample_rate, audio = wav_decoder.sample_rate.numpy(), wav_decoder.audio.numpy()
    return audio, sample_rate




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

        segments.append(segment)

    return segments


def segment_audio_return_rfft(audio_data, sample_rate, segment_length_secs):
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
        segments.append(rfft(segment))

    return segments



def segment_audio_return_mfcc(audio_data, sample_rate, segment_length_secs):
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

        # Compute MFCCs
        n_mfcc = 100  # Number of MFCC coefficients (you can adjust this)
        mfccs = librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=n_mfcc)


        segments.append(mfccs)

    return segments



def segment_audio_return_chroma_features(audio_data, sample_rate, segment_length_secs):
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

""" 
# Example usage
audio, sample_rate = read_file("example")
segment_length_secs = 10
segments = segment_audio(audio, sample_rate, segment_length_secs)
 """

def save_to_wav(segment_length_secs, original_segment, reconstructed_segment, sample_rate, output_folder, index):
    # Convert segments to time-domain signals
    # original_audio = np.array(original_segment).flatten()
    original_audio = np.array(irfft(original_segment)).flatten()
   
    # reconstructed_audio = np.array(reconstructed_segment).flatten()
    reconstructed_audio = np.array(irfft(reconstructed_segment)).flatten()
    
    # Calculate the number of samples in the segment
    num_samples = len(original_audio)
    num_samples1 = len(reconstructed_audio)
    
    # Create the time array based on the sample rate and number of samples
    t = np.linspace(0, segment_length_secs * num_samples, num_samples)
    t1 = np.linspace(0, segment_length_secs * num_samples1, num_samples1)
    
    # Plot and save the original segment
    plt.figure(figsize=(8, 4))
    plt.plot(t, original_audio)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Original Segment')
    plt.savefig(f'{output_folder}/original_segment_{index}.png')
    plt.close()
    
    # Plot and save the reconstructed segment
    plt.figure(figsize=(8, 4))
    plt.plot(t1, reconstructed_audio)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Reconstructed Segment')
    plt.savefig(f'{output_folder}/reconstructed_segment_{index}.png')
    plt.close()
    
    # Save the original and reconstructed segments as WAV files
    wavfile.write(f'{output_folder}/original_segment_{index}.wav', sample_rate, original_audio)
    wavfile.write(f'{output_folder}/reconstructed_segment_{index}.wav', sample_rate, reconstructed_audio)


def convert_mp3_to_wav():
    index = 0
    print("Convert")
    for file in iglob(DATA_FILES_MP3 + "/*.mp3"):
        print(f"index: {index}")
        mp3_to_wav = AudioSegment.from_mp3(file)
        mp3_to_wav.export(DATA_FILES_WAV + "/" + str(index) 
                          + ".wav", format = "wav")
        index += 1


# %%
