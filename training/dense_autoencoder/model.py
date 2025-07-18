
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_dense_autoencoder(input_dim=256, compressed_size=32):
    """
    Create dense autoencoder for hidden feature vectors.
    
    Args:
        input_dim: Dimension of input feature vectors
        compressed_size: Size of compressed representation
    """
    
    # Encoder
    encoder_input = layers.Input(shape=(input_dim,), name='encoder_input')
    
    # Encoder layers
    x = layers.Dense(128, activation='relu')(encoder_input)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    encoded = layers.Dense(compressed_size, activation='relu', name='encoded')(x)
    
    # Decoder
    x = layers.Dense(64, activation='relu')(encoded)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    # Output layer
    decoded = layers.Dense(input_dim, activation='linear', name='decoded')(x)
    
    # Create autoencoder
    autoencoder = keras.Model(encoder_input, decoded, name='dense_autoencoder')
    
    # Create encoder model for extracting compressed features
    encoder = keras.Model(encoder_input, encoded, name='dense_encoder')
    
    return autoencoder, encoder

def compile_model(autoencoder, learning_rate=0.001):
    """Compile the autoencoder model."""
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    return autoencoder
