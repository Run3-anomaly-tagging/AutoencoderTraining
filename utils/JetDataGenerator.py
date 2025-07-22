from tensorflow.keras.utils import Sequence
import numpy as np
import h5py
from AutoencoderTraining.utils.h5_helpers import extract_hidden_features, extract_jet_images

class JetDataGenerator(Sequence):
    def __init__(self, filepath, batch_size=256, mode='image', shuffle=True, indices=None):
        """
        Args:
            filepath (str): Path to HDF5 file containing 'Jets'
            batch_size (int): Number of samples per batch
            mode (str): 'image' or 'hidden'
            shuffle (bool): Whether to shuffle indices each epoch
        """
        self.filepath = filepath
        self.batch_size = batch_size
        self.mode = mode
        self.shuffle = shuffle
        self.file = None
    
        with h5py.File(self.filepath, 'r') as f:
            self.n_samples = len(f['Jets'])

        if indices is not None:
            self.indices = np.array(indices)
            self.n_samples = len(self.indices)
        else:
            self.indices = np.arange(self.n_samples)
        
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.ceil(self.n_samples / self.batch_size))
    
    def __getitem__(self, index):
        if self.file is None:
            self.file = h5py.File(self.filepath, 'r')

        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        with h5py.File(self.filepath, 'r') as f:
            jets = f['Jets'][sorted(batch_indices)]

        if self.mode == 'image':
            x = extract_jet_images(jets).astype(np.float32)
            x = np.expand_dims(x, -1)  # add channel dimension
        elif self.mode == 'hidden':
            x = extract_hidden_features(jets).astype(np.float32)
        else:
            raise ValueError(f"Unknown mode '{self.mode}'. Use 'image' or 'hidden'.")
        
        return x, x  # Autoencoder: input == output

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
