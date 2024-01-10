# %%
import os

# Change the current working directory
#new_directory = "/Users/michael/Uni/neural_networks/audio_convolution"
#os.chdir(new_directory)
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
print(f"current working directory: {os.getcwd()}")
from classifier import MusicClassifier
from classifier_dataset import LabeledMusicDataset
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



songs = []

for song_id in range(number_of_training_songs + number_of_validation_songs + number_of_testing_songs):
    print(f"Reading song {song_id}")
    songs.append(read_file(song_id))

print(f"Song count: {len(songs)}")


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

print(f"train_data shape: {train_data[0].shape}")

labeled_data = {'train': [], 'val': [], 'test': []}

# Iterate over the datasets: train_data, validation_data, and test_data
for key, dataset in zip(['train', 'val', 'test'], [train_data, validation_data, test_data]):
    # Label the original tensors with label 1
    for tensor in dataset:
        labeled_data[key].append((tensor.unsqueeze(2), 1))  # Add labeled tensor to corresponding dataset

    # Generate 1200 tensors of white noise with the same shape [210, 210]
    white_noise_tensors = [torch.randn(INPUT_DIMENSION, INPUT_DIMENSION).unsqueeze(2) for _ in range(len(dataset))]

    # Label the white noise tensors with label 0
    for tensor in white_noise_tensors:
        labeled_data[key].append((tensor, 0))  # Add labeled tensor to corresponding dataset

# Initialize the model, loss function, and optimizer
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 10

# Initialize the model, loss function, and optimizer
model = MusicClassifier(input_dimensions=INPUT_DIMENSION).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Prepare datasets and dataloaders
training_dataset = LabeledMusicDataset(labeled_data['train'])
train_loader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)

validation_dataset = LabeledMusicDataset(labeled_data['val'])
val_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

test_dataset = LabeledMusicDataset(labeled_data['test'])
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# %% Training loop
best_val_loss = float('inf')  # Initialize with a large value or use validation loss for comparison

for epoch in range(NUM_EPOCHS):
    # Training Phase
    model.train()  # Set model to training mode
    total_train_loss = 0.0
    
    loop = tqdm(enumerate(train_loader), total=len(train_loader))
    
    for batch_idx, (data, target) in loop:
        data, target = data.to(DEVICE), target.to(DEVICE)
        target = target.float()
        data = torch.tensor(data, dtype=torch.float32).to(DEVICE).view(data.shape[0], 1, INPUT_DIMENSION, INPUT_DIMENSION)
        optimizer.zero_grad()
        output = model(data)
        output = output.squeeze(1)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
        loop.set_description(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        loop.set_postfix(loss=total_train_loss / (batch_idx + 1))
    
    avg_train_loss = total_train_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Training Loss: {avg_train_loss:.4f}")

    # Validation Phase
    model.eval()
    total_val_loss = 0.0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            target = target.float()
            data = torch.tensor(data, dtype=torch.float32).to(DEVICE).view(data.shape[0], 1, INPUT_DIMENSION, INPUT_DIMENSION)
            output = model(data)
            output = output.squeeze(1)
            loss = criterion(output, target)
            total_val_loss += loss.item()
    
    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Validation Loss: {avg_val_loss:.4f}")
    
    # Check for improvement in validation loss and save the model if needed
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_model.pth')



# %% Set the model to evaluation mode
from sklearn.metrics import accuracy_score

# Set the model to evaluation mode
model.eval()

all_predictions = []
all_targets = []

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        
        # Ensure the data has the correct shape
        data = data.view(data.shape[0], 1, INPUT_DIMENSION, INPUT_DIMENSION)

        raw_scores = model(data)
        probabilities = torch.sigmoid(raw_scores)
        
        # Convert probabilities to binary predictions (0 or 1)
        predictions = (probabilities > 0.5).float()
        
        # Collect predictions and targets for later evaluation
        all_predictions.append(predictions.cpu().numpy())
        all_targets.append(target.cpu().numpy())

# Concatenate predictions and targets
all_predictions = np.concatenate(all_predictions)
all_targets = np.concatenate(all_targets)

# Calculate accuracy using scikit-learn
accuracy = accuracy_score(all_targets, all_predictions)
print(f"Accuracy on test data: {accuracy * 100:.2f}%")
# %%
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(all_targets, all_predictions)
print(cm)

from sklearn.metrics import classification_report
print(classification_report(all_targets, all_predictions))

from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(all_targets, all_predictions)
mse = mean_squared_error(all_targets, all_predictions)
print(mae)
print(mse)
# %%
from sklearn.metrics import precision_recall_curve
precision, recall, _ = precision_recall_curve(all_targets, all_predictions)
print(precision)
print(recall)
# %%   
