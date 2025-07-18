
import numpy as np
import tensorflow as tf
from tensorflow import keras
import h5py
import os
import matplotlib.pyplot as plt
from model import create_dense_autoencoder, compile_model
from utils.h5_helpers import extract_hidden_features

class DenseAutoencoderTrainer:
    def __init__(self, 
                 data_path: str = "data/merged/merged_qcd_train.h5",
                 model_save_path: str = "models/dense_autoencoder",
                 compressed_size: int = 32):
        
        self.data_path = data_path
        self.model_save_path = model_save_path
        self.compressed_size = compressed_size
        
        # Create model save directory
        os.makedirs(self.model_save_path, exist_ok=True)
    
    def load_data(self):
        """Load and preprocess hidden feature data."""
        print("Loading hidden feature data...")
        
        with h5py.File(self.data_path, 'r') as f:
            jets = f['Jets'][:]
        
        # Extract hidden features
        hidden_features = extract_hidden_features(jets)
        
        # Normalize features
        hidden_features = hidden_features.astype(np.float32)
        
        # Standardize features (zero mean, unit variance)
        mean = np.mean(hidden_features, axis=0)
        std = np.std(hidden_features, axis=0)
        std[std == 0] = 1  # Avoid division by zero
        
        hidden_features = (hidden_features - mean) / std
        
        print(f"Loaded {len(hidden_features)} feature vectors")
        print(f"Feature dimension: {hidden_features.shape[1]}")
        print(f"Feature range: [{hidden_features.min():.3f}, {hidden_features.max():.3f}]")
        
        # Store normalization parameters
        self.mean = mean
        self.std = std
        
        return hidden_features
    
    def split_data(self, hidden_features, validation_split=0.2):
        """Split data into training and validation sets."""
        n_samples = len(hidden_features)
        n_val = int(n_samples * validation_split)
        
        # Random split
        indices = np.random.permutation(n_samples)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        x_train = hidden_features[train_indices]
        x_val = hidden_features[val_indices]
        
        print(f"Training samples: {len(x_train)}")
        print(f"Validation samples: {len(x_val)}")
        
        return x_train, x_val
    
    def train(self, 
              epochs: int = 100,
              batch_size: int = 512,
              validation_split: float = 0.2,
              learning_rate: float = 0.001):
        """Train the dense autoencoder."""
        
        # Load data
        hidden_features = self.load_data()
        x_train, x_val = self.split_data(hidden_features, validation_split)
        
        # Create model
        print("Creating dense autoencoder model...")
        autoencoder, encoder = create_dense_autoencoder(
            input_dim=x_train.shape[1],
            compressed_size=self.compressed_size
        )
        
        # Compile model
        autoencoder = compile_model(autoencoder, learning_rate)
        
        # Print model summary
        print("\nAutoencoder Architecture:")
        autoencoder.summary()
        
        # Callbacks
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
            x_train, x_train,  # Autoencoder: input = output
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_val, x_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model and encoder
        autoencoder.save(os.path.join(self.model_save_path, 'final_autoencoder.h5'))
        encoder.save(os.path.join(self.model_save_path, 'final_encoder.h5'))
        
        # Save normalization parameters
        np.savez(os.path.join(self.model_save_path, 'normalization_params.npz'),
                 mean=self.mean, std=self.std)
        
        # Plot training history
        self.plot_training_history(history)
        
        print(f"Training complete! Models saved to {self.model_save_path}")
        
        return autoencoder, encoder, history
    
    def plot_training_history(self, history):
        """Plot training history."""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
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
        epochs=100,
        batch_size=512,
        learning_rate=0.001
    )