
import numpy as np
import tensorflow as tf
from tensorflow import keras
import h5py
import os
import matplotlib.pyplot as plt
from model import create_dense_autoencoder, compile_model
from AutoencoderTraining.utils.h5_helpers import extract_hidden_features
from AutoencoderTraining.paths import DEFAULT_MERGED_QCD_FILE, MODELS_DIR
from AutoencoderTraining.utils.JetDataGenerator import JetDataGenerator

class DenseAutoencoderTrainer:
    def __init__(self, 
                 data_path: str = DEFAULT_MERGED_QCD_FILE,
                 model_save_path: str = os.path.join(MODELS_DIR,"dense_autoencoder"),
                 compressed_size: int = 32):
        
        self.data_path = data_path
        self.model_save_path = model_save_path
        self.compressed_size = compressed_size
        
        # Create model save directory
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
            mode='hidden'
        )
        
        val_generator = JetDataGenerator(
            self.data_path,
            indices=val_indices,
            batch_size=batch_size,
            shuffle=False,
            mode='hidden'
        )

        return train_generator, val_generator
    
    def train(self, 
            epochs: int = 100,
            batch_size: int = 512,
            validation_split: float = 0.2,
            learning_rate: float = 0.001):
        """Train the dense autoencoder."""
        
        train_generator, val_generator = self.load_data(validation_split, batch_size)
        
        print("Creating dense autoencoder model...")
        autoencoder, encoder = create_dense_autoencoder(
            input_dim=train_generator[0][0].shape[1],  # get input_dim from generator batch shape
            compressed_size=self.compressed_size
        )
        
        autoencoder = compile_model(autoencoder, learning_rate)
        
        print("\nAutoencoder Architecture:")
        autoencoder.summary()
        
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.model_save_path, 'best_model.keras'),
                save_best_only=True,
                monitor='val_loss',
                mode='min',
                verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train model
        print("Starting training...")
        history = autoencoder.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        autoencoder.save(os.path.join(self.model_save_path, 'final_autoencoder.keras'))
        
        self.plot_training_history(history)
        
        print(f"Training complete! Models saved to {self.model_save_path}")
        
        return autoencoder, encoder, history

    
    def plot_training_history(self, history):
        """Plot training history."""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'][1:], label='Training Loss')#Skip the first epoch in plotting
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
    trainer = DenseAutoencoderTrainer()
    autoencoder, encoder, history = trainer.train(
        epochs=10,
        batch_size=512,
        learning_rate=0.001
    )