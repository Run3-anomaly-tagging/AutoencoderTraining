
import numpy as np
import tensorflow as tf
from tensorflow import keras
import h5py
import os
import matplotlib.pyplot as plt
from AutoencoderTraining.training.image_autoencoder.model import create_image_autoencoder, compile_model
from AutoencoderTraining.paths import DEFAULT_MERGED_QCD_FILE, MODELS_DIR
from AutoencoderTraining.utils.JetDataGenerator import JetDataGenerator

class ImageAutoencoderTrainer:
    def __init__(self, 
                 data_path: str = DEFAULT_MERGED_QCD_FILE,
                 model_save_path: str = os.path.join(MODELS_DIR,"image_autoencoder"),
                 compressed_size: int = 6):
        
        self.data_path = data_path
        self.model_save_path = model_save_path
        self.compressed_size = compressed_size
        
        os.makedirs(self.model_save_path, exist_ok=True)
        
    def load_data(self, validation_split=0.2, batch_size=256):
        """Prepare train and validation data generators."""
        print("Preparing data generators...")

        # Open the file once to get total number of jets
        with h5py.File(self.data_path, 'r') as f:
            total_samples = len(f['Jets'])
        
        # Create shuffled indices
        indices = np.random.permutation(total_samples)
        n_val = int(total_samples * validation_split)
        
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        print(f"Total samples: {total_samples}")
        print(f"Training samples: {len(train_indices)}")
        print(f"Validation samples: {len(val_indices)}")

        # Create generators for train and validation
        train_generator = JetDataGenerator(
            self.data_path,
            indices=train_indices,
            batch_size=batch_size,
            shuffle=True,
            mode='image'
        )
        
        val_generator = JetDataGenerator(
            self.data_path,
            indices=val_indices,
            batch_size=batch_size,
            shuffle=False,
            mode='image'
        )

        return train_generator, val_generator

    def train(self, epochs=2, batch_size=256, validation_split=0.2, learning_rate=0.001):

        print("TensorFlow version:", tf.__version__)
        print("Keras version:", keras.__version__)
        print("Available devices:")
        for device in tf.config.list_physical_devices():
            print(f"  - {device.device_type}: {device.name}")

        train_generator, val_generator = self.load_data(validation_split, batch_size)
        
        print("Creating image autoencoder model...")
        # Use input shape from the generator's output shape (example: first batch)
        sample_batch = train_generator[0][0]  # inputs from first batch
        autoencoder, encoder = create_image_autoencoder(
            input_shape=sample_batch.shape[1:],
            compressed_size=self.compressed_size
        )
        
        autoencoder = compile_model(autoencoder, learning_rate)
        
        print("\nAutoencoder Architecture:")
        autoencoder.summary()
        
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.model_save_path, 'best_model.h5'),
                save_best_only=True,
                monitor='val_loss',
                mode='min',
                verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        print("Starting training...")
        history = autoencoder.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        autoencoder.save(os.path.join(self.model_save_path, 'final_autoencoder.h5'))
        
        self.plot_training_history(history)
        
        print(f"Training complete! Models saved to {self.model_save_path}")
        
        return autoencoder, encoder, history

    def plot_training_history(self, history):
        """Plot training history."""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'][1:], label='Training Loss') #Skip plotting the first epoch
        plt.plot(history.history['val_loss'][1:], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'][1:], label='Training MAE')
        plt.plot(history.history['val_mae'][1:], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_save_path, 'training_history.png'))
        plt.show()

if __name__ == "__main__":
    trainer = ImageAutoencoderTrainer()
    autoencoder, encoder, history = trainer.train(
        epochs=2,
        batch_size=256,
        learning_rate=0.001
    )
