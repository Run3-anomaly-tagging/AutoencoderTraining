
import json
import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from utils.h5_helpers import count_jets_in_file

def explore_files(config_path: str = "configs/dataset_config.json"):
    """Explore and count jets in all configured files."""
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("File exploration results:")
    print("=" * 50)
    
    total_qcd_jets = 0
    for name, info in config['samples'].items():
        filepath = info['path']
        xsec = info['xsec']
        
        if os.path.exists(filepath):
            count = count_jets_in_file(filepath)
            total_qcd_jets += count
            print(f"{name:20} | {count:8d} jets | xsec: {xsec:.2e}")
        else:
            print(f"{name:20} | FILE NOT FOUND")
    
    print("-" * 50)
    print(f"Total QCD jets: {total_qcd_jets:,}")
    
    # Check signal sample
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
