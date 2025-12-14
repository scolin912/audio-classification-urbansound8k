import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa

# ---- Feature config (must match training) ----
SR = 22050
TARGET_SEC = 4.0
N_MELS = 64
FMAX = 8000

def load_audio_fixed(path: Path, sr=SR, target_sec=TARGET_SEC):
    y, _sr = librosa.load(str(path), sr=sr, mono=True)  # supports wav/mp3
    target_len = int(sr * target_sec)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y

def wav_to_logmel(path: Path):
    y = load_audio_fixed(path)
    S = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, fmax=FMAX)
    logS = librosa.power_to_db(S, ref=np.max)
    logS = (logS - logS.mean()) / (logS.std() + 1e-6)
    return logS.astype(np.float32)  # (mels, frames)

# ---- Model (same as notebook, auto-flat) ----
class SmallCNN(nn.Module):
    def __init__(self, num_classes, input_shape):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        self.pool  = nn.MaxPool2d(2,2)
        self.drop  = nn.Dropout(0.3)

        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            z = self._features(dummy)
            self.flat_dim = z.view(1, -1).shape[1]

        self.fc1 = nn.Linear(self.flat_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def _features(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        return x

    def forward(self, x):
        x = self._features(x)
        x = self.drop(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)

def pretty_topk(classes, probs, k=5):
    top_idx = probs.argsort()[::-1][:k]
    for i in top_idx:
        p = float(probs[i])
        bar = "█" * int(p * 30)
        print(f"{classes[i]:18s} {p*100:6.2f}%  {bar}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio_path", type=str, help="Path to wav/mp3")
    ap.add_argument("--model", type=str, default="models/us8k_smallcnn.pth")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    audio_path = Path(args.audio_path)
    ckpt_path = Path(args.model)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Model not found: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    classes = ckpt["classes"]

    # build input shape dynamically from the file (matches training feature config)
    logS = wav_to_logmel(audio_path)
    input_shape = (1, logS.shape[0], logS.shape[1])

    model = SmallCNN(num_classes=len(classes), input_shape=input_shape)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    x = torch.tensor(logS).unsqueeze(0).unsqueeze(0)  # (1,1,mels,frames)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).numpy()

    pretty_topk(classes, probs, k=args.topk)

if __name__ == "__main__":
    main()
