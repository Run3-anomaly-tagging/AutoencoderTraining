import json
import os
import numpy as np
import h5py
from typing import Tuple
from AutoencoderTraining.utils.h5_helpers import load_jets_from_file, save_jets_to_file, count_jets_in_file
from AutoencoderTraining.paths import DEFAULT_CONFIG_PATH


class JetSampler:
    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.samples = self.config['samples']
        self.sampling_strategy = self.config['sampling_strategy']
        self.max_jets_per_file = self.config['max_jets_per_file']
        self.output_file = self.config['output_file']
    
    def bin_normalized_sampling(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bin-normalized sampling strategy:
        - Load all jets from the lowest HT bin (N)
        - Load up to M = xsec_ratio * N jets from each higher bin
        """
        print("Using bin-normalized sampling strategy...")
        
        # Sort samples by cross-section (descending = lowest HT first)
        sorted_samples = sorted(self.samples.items(), 
                                key=lambda x: x[1]['xsec'], 
                                reverse=True)
        
        # Get reference count from lowest HT bin (highest xsec)
        ref_name, ref_info = sorted_samples[0]
        ref_count = count_jets_in_file(ref_info['path'])
        ref_count = min(ref_count, self.max_jets_per_file)
        
        print(f"Reference sample: {ref_name} with {ref_count} jets")
        
        all_jets = []
        all_weights = []
        
        for name, info in sorted_samples:
            filepath = info['path']
            xsec = info['xsec']
            
            if name == ref_name:
                # Reference sample: load all (up to max)
                target_count = ref_count
            else:
                # Other samples: scale by cross-section ratio
                xsec_ratio = xsec / sorted_samples[0][1]['xsec']
                target_count = int(ref_count * xsec_ratio)
                target_count = max(1, target_count)  # At least 1 jet
            
            jets = load_jets_from_file(filepath, target_count)
            weights = np.ones(len(jets), dtype=np.float32)  # All weights = 1
            
            all_jets.append(jets)
            all_weights.append(weights)
            
            print(f"  {name:20} | {len(jets):6d} jets | target: {target_count:6d}")
        
        # Concatenate all jets
        merged_jets = np.concatenate(all_jets, axis=0)
        merged_weights = np.concatenate(all_weights, axis=0)
        
        # Shuffle the combined dataset
        indices = np.random.permutation(len(merged_jets))
        merged_jets = merged_jets[indices]
        merged_weights = merged_weights[indices]
        
        return merged_jets, merged_weights
    
    def weighted_rescaled_sampling(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Weighted rescaled sampling strategy:
        - Load up to max_jets_per_file from each file
        - Assign weights based on cross-section ratios
        """
        print("Using weighted rescaled sampling strategy...")
        
        # Get reference cross-section (lowest HT bin = highest xsec)
        ref_xsec = max(info['xsec'] for info in self.samples.values())
        
        all_jets = []
        all_weights = []
        
        for name, info in self.samples.items():
            filepath = info['path']
            xsec = info['xsec']
            
            jets = load_jets_from_file(filepath, self.max_jets_per_file)
            
            # Calculate weights: weight = xsec_bin / num_loaded_bin, normalized to ref
            weight_per_jet = (xsec / len(jets)) / (ref_xsec / self.max_jets_per_file)
            weights = np.full(len(jets), weight_per_jet, dtype=np.float32)
            
            all_jets.append(jets)
            all_weights.append(weights)
            
            print(f"  {name:20} | {len(jets):6d} jets | weight: {weight_per_jet:.4f}")
        
        # Concatenate and shuffle
        merged_jets = np.concatenate(all_jets, axis=0)
        merged_weights = np.concatenate(all_weights, axis=0)
        
        indices = np.random.permutation(len(merged_jets))
        merged_jets = merged_jets[indices]
        merged_weights = merged_weights[indices]
        
        return merged_jets, merged_weights
    
    def sample_data(self) -> None:
        """
        Stream through each input file, sample its jets according to the chosen strategy,
        and append them (with weights) directly into the output HDF5 file.
        """
        print(f"Starting data sampling with strategy: {self.sampling_strategy}")

        # Remove existing output file if present
        try:
            os.remove(self.output_file)
        except FileNotFoundError:
            pass

        # Prepare iteration order and reference info
        items = list(self.samples.items())
        if self.sampling_strategy == "bin-normalized":
            # Sort by descending xsec so the first is the reference
            sorted_items = sorted(items, key=lambda x: x[1]['xsec'], reverse=True)
            ref_name, ref_info = sorted_items[0]
            ref_count = count_jets_in_file(ref_info['path'])
            ref_count = min(ref_count, self.max_jets_per_file)
            processing_list = sorted_items
        elif self.sampling_strategy == "weighted-rescaled":
            # For weights
            ref_xsec = max(info['xsec'] for info in self.samples.values())
            processing_list = items
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")

        # Open output HDF5 once
        with h5py.File(self.output_file, "w") as out:
            for name, info in processing_list:
                filepath = info['path']
                xsec = info['xsec']

                if self.sampling_strategy == "bin-normalized":
                    if name == ref_name:
                        target_count = ref_count
                    else:
                        ratio = xsec / ref_info['xsec']
                        target_count = max(1, int(ref_count * ratio))
                    jets = load_jets_from_file(filepath, target_count)
                    weights = np.ones(len(jets), dtype=np.float32)
                else:  # weighted-rescaled
                    jets = load_jets_from_file(filepath, self.max_jets_per_file)
                    weight_per_jet = (xsec / len(jets)) / (ref_xsec / self.max_jets_per_file)
                    weights = np.full(len(jets), weight_per_jet, dtype=np.float32)

                print(f"  Appending {len(jets):6d} jets from {name}")

                # Create or append datasets
                if "jets" not in out:
                    maxshape = (None,) + jets.shape[1:]
                    out.create_dataset("jets", data=jets, maxshape=maxshape, chunks=True)
                    out.create_dataset("weights", data=weights, maxshape=(None,), chunks=True)
                else:
                    j_ds = out["jets"]
                    w_ds = out["weights"]
                    old_size = j_ds.shape[0]
                    new_size = old_size + jets.shape[0]
                    j_ds.resize((new_size,) + jets.shape[1:])
                    j_ds[old_size:new_size] = jets
                    w_ds.resize((new_size,))
                    w_ds[old_size:new_size] = weights

        print("All samples processed and appended. Sampling complete!")


if __name__ == "__main__":
    sampler = JetSampler()
    sampler.sample_data()
