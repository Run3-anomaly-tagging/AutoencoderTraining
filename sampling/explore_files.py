import json
import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parent_dir = os.path.dirname(project_root)
sys.path.insert(0, parent_dir)
from AutoencoderTraining.utils.h5_helpers import count_jets_in_file

config_path_abs = os.path.join(project_root,"configs/dataset_config.json")
print("DEBUG: ", config_path_abs)

def explore_files(config_path: str = config_path_abs):
    """Explore and count jets in all configured files, and scale to first non-empty HT bin."""
    with open(config_path, 'r') as f:
        config = json.load(f)

    print("File exploration results:")
    print("=" * 50)

    total_qcd_jets = 0
    total_scaled_jets = 0
    reference_count = None
    reference_xsec = None
    reference_bin = None

    for name, info in config['samples'].items():
        filepath = info['path']
        xsec = info['xsec']

        if os.path.exists(filepath):
            count = count_jets_in_file(filepath)
            total_qcd_jets += count

            # Use first valid bin (non-zero jets) as reference
            if reference_count is None and count > 0:
                reference_count = count
                reference_xsec = xsec
                reference_bin = name

            scale = int((xsec / reference_xsec) * reference_count) if reference_count else 0
            total_scaled_jets += scale
            print(f"{name:20} | {count:8d} jets | xsec: {xsec:.2e} | scaled max: {scale}")
        else:
            print(f"{name:20} | FILE NOT FOUND")

    print("-" * 50)
    print(f"Reference bin: {reference_bin} ({reference_count} jets)")
    print(f"Total QCD jets: {total_qcd_jets:,}")
    print(f"Total QCD jets after sampling: {total_scaled_jets:,}")

    # Signal sample
    signal_info = config['signal_sample']
    for name, info in signal_info.items():
        filepath = info['path']
        if os.path.exists(filepath):
            count = count_jets_in_file(filepath)
            print(f"{name:20} | {count:8d} jets (signal)")
        else:
            print(f"{name:20} | FILE NOT FOUND (signal)")


if __name__ == "__main__":
    explore_files()
