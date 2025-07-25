import tensorflow as tf
import numpy as np
import h5py
from sklearn.metrics import roc_auc_score


class AUCMetricCallback(tf.keras.callbacks.Callback):
    """
    Computes AUC on a dataset feature after each epoch.

    Args:
        auc_dataset_path (str): Path to AUC dataset.
        feature_key (str): Feature to evaluate ('hidNeurons' or 'jet_image').
        batch_size (int): Batch size for prediction (default 1024).
        name (str): Metric name for logging (default 'val_auc').
    """
    def __init__(self, auc_dataset_path, feature_key, batch_size=1024,name="val_auc"):
        super().__init__()
        self.auc_dataset_path = auc_dataset_path
        self.batch_size = batch_size
        self.metric_name = name
        self.feature_key = feature_key

    def _load_jets(self):
        """Load signal and background jets from the AUC validation file."""
        with h5py.File(self.auc_dataset_path, 'r') as f:
            bkg_jets = f['Jets_Bkg'][:]
            sig_jets = f['Jets_Signal'][:]

        x = np.concatenate([bkg_jets, sig_jets], axis=0)
        y = np.array([0] * len(bkg_jets) + [1] * len(sig_jets))
        return x, y

    def on_epoch_end(self, epoch, logs=None):
        x, y_true = self._load_jets()

        # Predict in batches
        preds = []
        for i in range(0, len(x), self.batch_size):
            batch = x[i:i + self.batch_size]
            batch_features = batch[self.feature_key]
            if batch_features.ndim == 3:  # (batch_size, H, W), add missing channel dim in image input
                batch_features = np.expand_dims(batch_features, axis=-1)
            recon = self.model.predict(batch_features, verbose=0)
            errors = np.mean(np.square(batch_features - recon), axis=tuple(range(1, batch_features.ndim)))

            preds.extend(errors)

        preds = np.array(preds)

        auc = roc_auc_score(y_true, preds)
        print(f"\nEpoch {epoch + 1} — ROC AUC (QCD vs Signal): {auc:.4f}")
        if logs is not None:
            logs[self.metric_name] = auc
        else:
            self.model.history.history.setdefault(self.metric_name, []).append(auc)
