from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, Sequential

NUMBER_OF_FIRST_HIDDEN_LAYER_NODES = 8400
NUMBER_OF_SECOND_HIDDEN_LAYER_NODES = 3440
LATENT_SPACE_DIMENSIONS = 2800

class Autoencoder(Model):
    def __init__(self, input_length):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = Sequential([
            Input(shape=(input_length,)),  # Input layer
            Dense(NUMBER_OF_FIRST_HIDDEN_LAYER_NODES, activation='relu'),  # First encoder layer
            Dense(NUMBER_OF_SECOND_HIDDEN_LAYER_NODES, activation='relu'),   # Second encoder layer
            Dense(LATENT_SPACE_DIMENSIONS, activation='relu')  # Latent space layer
        ])

        # Decoder
        self.decoder = Sequential([
            Input(shape=(LATENT_SPACE_DIMENSIONS,)),  # Input layer for decoder
            Dense(NUMBER_OF_SECOND_HIDDEN_LAYER_NODES, activation='relu'),  # Second decoder layer
            Dense(NUMBER_OF_FIRST_HIDDEN_LAYER_NODES, activation='relu'),  # Second decoder layer
            Dense(input_length, activation=None)  # Output layer
        ])

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

""" 
# Create an instance of the Autoencoder
autoencoder = Autoencoder(latent_space_dimensions=32, input_length=256)
 """