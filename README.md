# UrbanSound8K Audio Classifier (Log-Mel + CNN)

An end-to-end audio classification project built with **PyTorch** using the **UrbanSound8K** dataset.  
The system supports **training in Jupyter Notebook** and **command-line inference** on `.wav` and `.mp3` files.

---

## 🔍 Project Overview

- **Task**: Environmental sound classification (10 classes)
- **Dataset**: UrbanSound8K
- **Features**: Log-mel spectrogram (fixed 4 seconds)
- **Model**: Lightweight CNN (SmallCNN)
- **Framework**: PyTorch
- **Inference**: CLI (`predict.py`) supporting `.wav` / `.mp3`

---

## 📊 Results

- **Test Accuracy (fold10)**: **~0.705**
- Training setup:
  - Train folds: 1–8
  - Validation fold: 9
  - Test fold: 10
  - Epochs: 5
  - Optimizer: Adam (lr=1e-3)

##Example inference output:
```text
dog_bark            99.94%  █████████████████████████████
children_playing     0.03%
drilling             0.02%
gun_shot             0.00%
siren                0.00%



## 📁 Project Structure
urbansound8k_baseline/
├── notebooks/
│ └── train.ipynb # Training & experiments (Jupyter)
├── models/
│ └── us8k_smallcnn.pth # Trained model checkpoint
├── predict.py # CLI inference script
├── requirements.txt
└── README.md

## ⚙️ Environment Setup

Recommended environment (conda):

```bash
conda create -n audio2 python=3.10 -y
conda activate audio2
pip install -r requirements.txt


### ▶️ Usage

#### Inference (Command Line)

Once the trained model checkpoint is available (`models/us8k_smallcnn.pth`),
you can run inference on any audio file without moving it into the project directory.

Both absolute and relative paths are supported.

```bash
conda activate audio2

# absolute path
python predict.py examples/audio.mp3 --topk 3


# relative path
python predict.py ../samples/audio.wav --topk 3

Supported formats:
.wav
.mp3

