import numpy as np
import matplotlib.pyplot as plt
import h5py
from tensorflow import keras
from AutoencoderTraining.paths import DEFAULT_MERGED_QCD_FILE, DEFAULT_SIGNAL_FILE
from AutoencoderTraining.paths import DENSE_AUTOENCODER_MODEL, IMAGE_AUTOENCODER_MODEL
from AutoencoderTraining.utils.h5_helpers import extract_hidden_features, extract_jet_images
import os
from sklearn.metrics import roc_curve, auc

N_PLOT=100000

def load_data(file_path, key='Jets', extractor=None):
    with h5py.File(file_path, 'r') as f:
        jets = f[key][:N_PLOT]
    if extractor is not None:
        return extractor(jets)
    else:
        return jets

def get_reco_errors(x, model, batch_size=1024):
    preds = model.predict(x, batch_size=batch_size, verbose=0)
    errors = np.mean((x - preds) ** 2, axis=tuple(range(1, x.ndim)))  # MSE per sample
    return errors


for MODEL_PATH, extractor in zip([DENSE_AUTOENCODER_MODEL, IMAGE_AUTOENCODER_MODEL],[extract_hidden_features, extract_jet_images]):    
    model_name = os.path.basename(os.path.dirname(MODEL_PATH))  # "dense_autoencoder" or "image_autoencoder"
    autoencoder = keras.models.load_model(MODEL_PATH, compile=False)

    x_bkg = load_data(DEFAULT_MERGED_QCD_FILE, extractor=extractor)
    x_sig = load_data(DEFAULT_SIGNAL_FILE, extractor=extractor)

    if x_bkg.ndim == 3:
        x_bkg = np.expand_dims(x_bkg, -1)
    if x_sig.ndim == 3:
        x_sig = np.expand_dims(x_sig, -1)

    print("Computing reconstruction errors...")
    reco_bkg = get_reco_errors(x_bkg, autoencoder)
    reco_sig = get_reco_errors(x_sig, autoencoder)

    plt.figure(figsize=(8,6))
    plt.hist(reco_bkg, bins=100, histtype='step', label='QCD (background)', density=True)
    plt.hist(reco_sig, bins=100, histtype='step', label='H->bb (signal)', density=True)
    plt.xlabel("Reconstruction MSE")
    plt.ylabel("Density")
    #plt.yscale("log")
    plt.legend()
    plt.tight_layout()

    out_path = f"evaluation/reco_loss_{model_name}.png"
    plt.savefig(out_path)
    print(f"Saved: {out_path}")


    if model_name == "dense_autoencoder":#AUC needs to be implemented in image_ae first

        y_true = np.concatenate([np.zeros_like(reco_bkg), np.ones_like(reco_sig)])
        y_scores = np.concatenate([reco_bkg, reco_sig])

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc_val = auc(fpr, tpr)

        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f"AUC = {auc_val:.3f}")
        plt.xlabel("Background mistag rate")
        plt.ylabel("Signal efficiency")
        plt.title("ROC Curve (Dense Autoencoder)")
        plt.legend(loc="lower right")
        plt.tight_layout()

        out_path = "evaluation/roc_dense_autoencoder.png"
        plt.savefig(out_path)
        print(f"Saved: {out_path}")
