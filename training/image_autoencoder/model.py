
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

def create_image_autoencoder(input_shape=(32, 32, 1), compressed_size=6):
    npix = input_shape[0]
    mini_size = npix // 4

    encoder_input = layers.Input(shape=input_shape, name='encoder_input')

    # Encoder
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(encoder_input)
    x = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    
    x = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    
    x = layers.Conv2D(4, (3, 3), padding='same', activation='relu')(x)

    x = layers.Flatten()(x)
    x = layers.Dense(16, activation='relu')(x)

    # Compressed (latent) layer
    encoded = layers.Dense(compressed_size, activation='relu', name='encoded')(x)

    # Decoder
    x = layers.Dense(16, activation='relu')(encoded)
    x = layers.Dense((mini_size * mini_size) * 4, activation='relu')(x)
    
    x = layers.Reshape((mini_size, mini_size, 4))(x)

    x = layers.Conv2D(4, (3, 3), padding='same', activation='relu')(x)
    x = layers.UpSampling2D((2, 2))(x)
    
    x = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2D(1, (3, 3), padding='same')(x)
    x = layers.Reshape((1, npix * npix))(x)
    x = layers.Activation('softmax')(x)
    decoded = layers.Reshape((npix, npix, 1), name='decoded')(x)

    autoencoder = Model(encoder_input, decoded, name='image_autoencoder')
    encoder = Model(encoder_input, encoded, name='image_encoder')

    return autoencoder, encoder

def compile_model(autoencoder, learning_rate=0.001):
    """Compile the autoencoder model."""
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    return autoencoder