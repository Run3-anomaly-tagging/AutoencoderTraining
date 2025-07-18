
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_image_autoencoder(input_shape=(32, 32, 1), compressed_size=6):
    """
    Create convolutional autoencoder for jet images.
    
    Args:
        input_shape: Shape of input jet images
        compressed_size: Size of compressed representation
    """
    
    # Encoder
    encoder_input = layers.Input(shape=input_shape, name='encoder_input')
    
    # Encoder layers
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_input)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    
    # Flatten and compress
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    encoded = layers.Dense(compressed_size, activation='relu', name='encoded')(x)
    
    # Decoder
    x = layers.Dense(64, activation='relu')(encoded)
    x = layers.Dense(4 * 4 * 128, activation='relu')(x)
    x = layers.Reshape((4, 4, 128))(x)
    
    # Decoder layers
    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    
    # Output layer
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='decoded')(x)
    
    # Create autoencoder
    autoencoder = keras.Model(encoder_input, decoded, name='image_autoencoder')
    
    # Create encoder model for extracting compressed features
    encoder = keras.Model(encoder_input, encoded, name='image_encoder')
    
    return autoencoder, encoder

def compile_model(autoencoder, learning_rate=0.001):
    """Compile the autoencoder model."""
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    return autoencoder