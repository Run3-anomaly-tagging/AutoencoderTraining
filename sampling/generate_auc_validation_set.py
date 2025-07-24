import os
import json
import numpy as np
import h5py

from AutoencoderTraining.paths import DEFAULT_CONFIG_PATH, DEFAULT_MERGED_QCD_FILE
from AutoencoderTraining.utils.h5_helpers import load_jets_from_file


class AUCSetBuilder:
    def __init__(self, config_path=DEFAULT_CONFIG_PATH):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.auc_config = self.config['auc_set']
        self.signal_samples = self.config['signal_samples']
        
        self.qcd_source_path = DEFAULT_MERGED_QCD_FILE
        self.output_file = self.auc_config['output_file']
        self.n_qcd_jets = self.auc_config['n_qcd_jets']
        self.n_signal_jets = self.auc_config['n_signal_jets']
        self.selected_signals = self.auc_config['signal_samples']

    def load_qcd_jets(self):
        print(f"Loading {self.n_qcd_jets} QCD jets from {self.qcd_source_path}")
        return load_jets_from_file(self.qcd_source_path, self.n_qcd_jets)

    def load_signal_jets(self):
        jets = []
        jets_per_sample = self.n_signal_jets // len(self.selected_signals)

        print(f"Loading {self.n_signal_jets} signal jets from: {', '.join(self.selected_signals)}")
        for name in self.selected_signals:
            if name not in self.signal_samples:
                raise ValueError(f"Signal sample '{name}' not found in config.")
            
            path = self.signal_samples[name]['path']
            loaded = load_jets_from_file(path, jets_per_sample)
            jets.append(loaded)
            print(f"  Loaded {len(loaded)} jets from {name}")

        all_signal = np.concatenate(jets, axis=0)
        if len(all_signal) > self.n_signal_jets:
            all_signal = all_signal[:self.n_signal_jets]
        return all_signal

    def save_to_file(self, qcd_jets, signal_jets):
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        print(f"Saving to {self.output_file}...")
        with h5py.File(self.output_file, 'w') as f:
            f.create_dataset("Jets_Bkg", data=qcd_jets, compression='gzip')
            f.create_dataset("Jets_Signal", data=signal_jets, compression='gzip')
        print("Saved AUC validation dataset successfully.")

    def generate(self):
        if os.path.exists(self.output_file):
            print(f"AUC validation file already exists: {self.output_file} — skipping.")
            return

        qcd = self.load_qcd_jets()
        signal = self.load_signal_jets()
        self.save_to_file(qcd, signal)


if __name__ == "__main__":
    builder = AUCSetBuilder()
    builder.generate()
