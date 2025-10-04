import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa

# ----------------------------
# Reproducibility (optional)
# ----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ----------------------------
# Dataset
# ----------------------------
class RespiratorySoundDataset(Dataset):
    """
    Expects pairs of .wav and .txt files in the same directory:
      example.wav
      example.txt  (with lines containing ... crackle wheeze)
    Label is 1 if any line has crackle==1 OR wheeze==1, else 0.
    """
    def __init__(self, data_dir, sr=22050, duration=20, n_mfcc=40):
        self.data_dir = data_dir
        self.sr = sr
        self.duration = duration
        self.n_mfcc = n_mfcc
        self.wav_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.wav')]

    def __len__(self):
        return len(self.wav_files)

    def __getitem__(self, idx):
        wav_file = self.wav_files[idx]
        wav_path = os.path.join(self.data_dir, wav_file)
        txt_file = wav_file[:-4] + '.txt'
        txt_path = os.path.join(self.data_dir, txt_file)

        # Load and pad/trim audio to fixed duration
        audio, _ = librosa.load(wav_path, sr=self.sr)
        max_len = self.sr * self.duration
        if len(audio) < max_len:
            audio = np.pad(audio, (0, max_len - len(audio)), mode='constant')
        else:
            audio = audio[:max_len]

        # MFCCs (fixed params)
        mfcc = librosa.feature.mfcc(
            y=audio, sr=self.sr, n_mfcc=self.n_mfcc,
            n_fft=1024, hop_length=512, win_length=1024
        )
        # Normalize per-sample
        mean = np.mean(mfcc)
        std = np.std(mfcc) + 1e-6
        mfcc = (mfcc - mean) / std

        # (C, F, T) where C=1
        mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)

        # Read annotation for binary label
        label_val = 0
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 4:
                        crackle = int(parts[2])
                        wheeze = int(parts[3])
                        if crackle == 1 or wheeze == 1:
                            label_val = 1
                            break
        label = torch.tensor(label_val, dtype=torch.long)
        return mfcc_tensor, label

# ----------------------------
# Model
# ----------------------------
class LungSoundCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.dropout = nn.Dropout(0.3)
        self._to_linear = None
        self.fc1 = None
        self.fc2 = nn.Linear(64, 2)  # 0=Normal, 1=Abnormal

    def forward(self, x):
        # Create fc1 dynamically on first forward, based on actual input size
        if self._to_linear is None:
            self._to_linear = self._get_conv_output(x)
            self.fc1 = nn.Linear(self._to_linear, 64).to(x.device)

        x = self.convs(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

    def _get_conv_output(self, x):
        with torch.no_grad():
            x = self.convs(x)
            return int(np.prod(x.size()[1:]))

# ----------------------------
# Train
# ----------------------------
def train(data_dir, epochs=20, batch_size=16, lr=1e-3):
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = RespiratorySoundDataset(data_dir)
    # num_workers=0 for Windows is safest
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = LungSoundCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)  # first pass will also create fc1
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total if total else 0.0
        epoch_acc = correct / total if total else 0.0
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f}")

    # Save the entire model (architecture + weights)
    torch.save(model, "lung_sound_model_full.pth")
    print("Full model saved to lung_sound_model_full.pth")

# ----------------------------
# Entry
# ----------------------------
if __name__ == "__main__":
    data_folder_path = r"C:\Academic\robo games\git repo\NeuroLung\dat\data_set"  # update if needed
    train(data_folder_path)
