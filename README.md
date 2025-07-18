### 1. Setup
```bash
git clone git@github.com:Run3-anomaly-tagging/AutoencoderTraining.git
cd AutoencoderTraining
python3 -m venv venv
cd ..
echo $PWD > AutoencoderTraining/venv/lib/python3.9/site-packages/project_root.pth
cd AutoencoderTraining
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Config
Edit `configs/dataset_config.json` to match your data paths and cross-sections.

### 3. Check datasets
```bash
python sampling/explore_files.py
```

### 4. Create samples training dataset (HT merging)
```bash
python sampling/sample_strategy.py
```

### 5. Training

On lpc-gpu nodes needs to be done with apptainer

```bash
./run_training_image.sh #Seems to work, but gives a warning, to be checked
```
