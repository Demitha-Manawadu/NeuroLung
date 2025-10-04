import os
import numpy as np
import torch
import torch.nn as nn
import librosa
import tkinter as tk
from tkinter import filedialog, messagebox

# Optional drag & drop
USE_DND = False
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    USE_DND = True
except Exception:
    USE_DND = False

# ----------------------------
# Model architecture (same as training)
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
# Preprocessing (must match training)
# ----------------------------
def preprocess_wav(wav_path, sr=22050, duration=20, n_mfcc=40):
    audio, _ = librosa.load(wav_path, sr=sr)
    max_len = sr * duration
    if len(audio) < max_len:
        audio = np.pad(audio, (0, max_len - len(audio)), mode='constant')
    else:
        audio = audio[:max_len]

    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=n_mfcc,
        n_fft=1024, hop_length=512, win_length=1024
    )
    mean = np.mean(mfcc)
    std = np.std(mfcc) + 1e-6
    mfcc = (mfcc - mean) / std

    # (B=1, C=1, F, T)
    x = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return x

class_names = ["Normal", "Abnormal"]

# ----------------------------
# Robust loader that accepts full model OR state_dict
# ----------------------------
def try_load_model_or_state(path, device):
    """
    Returns (model, state_dict) where one of them is not None.
    - If file is a full nn.Module: (model, None)
    - If file is a state_dict: (None, state_dict)
    """
    obj = torch.load(path, map_location=device)
    if isinstance(obj, nn.Module):
        model = obj.to(device)
        model.eval()
        return model, None
    elif isinstance(obj, dict):
        return None, obj
    else:
        raise RuntimeError(f"Unsupported checkpoint type: {type(obj)}")

def build_model_for_input(x, device):
    """
    Creates LungSoundCNN and runs one dry forward to size fc1
    using the given preprocessed input x.
    """
    model = LungSoundCNN().to(device)
    with torch.no_grad():
        _ = model(x.to(device))  # creates fc1 with correct shape
    return model

# ----------------------------
# GUI App
# ----------------------------
class App:
    def __init__(self, master, device, ckpt_path="lung_sound_model_full.pth"):
        self.master = master
        self.device = device
        self.ckpt_path = ckpt_path

        master.title("Lung Sound Classifier (Normal vs Abnormal)")
        master.geometry("560x270")
        master.resizable(False, False)

        tk.Label(master, text="Lung Sound Classifier", font=("Segoe UI", 12)).pack(pady=10)

        if not os.path.exists(self.ckpt_path):
            messagebox.showerror("Error", f"Model file not found:\n{self.ckpt_path}")
            master.destroy()
            return

        try:
            # Try to read either a full model or a state_dict
            self.model, self.state_dict = try_load_model_or_state(self.ckpt_path, self.device)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model file:\n{e}")
            master.destroy()
            return

        # Drop target or instructions
        if USE_DND:
            self.drop_label = tk.Label(master, text="Drag & drop a .wav here or click 'Choose .wav'",
                                       relief="groove", width=62, height=4)
            self.drop_label.pack(pady=8)
            self.drop_label.drop_target_register(DND_FILES)
            self.drop_label.dnd_bind('<<Drop>>', self.on_drop)
        else:
            tk.Label(master, text="Click 'Choose .wav' to select an audio file.").pack(pady=8)

        # Buttons
        btn_frame = tk.Frame(master)
        btn_frame.pack(pady=4)
        tk.Button(btn_frame, text="Choose .wav", command=self.choose_wav).grid(row=0, column=0, padx=6)
        tk.Button(btn_frame, text="Quit", command=master.quit).grid(row=0, column=1, padx=6)

        # Results
        self.result_var = tk.StringVar(value="Prediction: —")
        self.score_var = tk.StringVar(value="Probability: —")
        tk.Label(master, textvariable=self.result_var, font=("Segoe UI", 11)).pack(pady=6)
        tk.Label(master, textvariable=self.score_var, font=("Segoe UI", 10)).pack()

        dev_name = "CUDA" if torch.cuda.is_available() else "CPU"
        tk.Label(master, text=f"Running on: {dev_name}").pack(pady=4)

    def on_drop(self, event):
        path = event.data.strip("{}")
        if os.path.isfile(path) and path.lower().endswith(".wav"):
            self.run_inference(path)
        else:
            messagebox.showwarning("Invalid file", "Please drop a single .wav file.")

    def choose_wav(self):
        path = filedialog.askopenfilename(title="Select .wav file", filetypes=[("WAV files", "*.wav")])
        if path:
            self.run_inference(path)

    def run_inference(self, wav_path):
        try:
            # Preprocess first
            x = preprocess_wav(wav_path).to(self.device)

            # If we only had a state_dict, build the model now with a dry forward to size fc1, then load weights
            if self.model is None and self.state_dict is not None:
                self.model = build_model_for_input(x, self.device)
                missing, unexpected = self.model.load_state_dict(self.state_dict, strict=False)
                # Optional: print mismatches for debugging
                # print("Missing:", missing, "Unexpected:", unexpected)
                self.model.eval()

            with torch.no_grad():
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                idx = int(np.argmax(probs))
                label = class_names[idx]
                self.result_var.set(f"Prediction: {label}")
                self.score_var.set(
                    f"Probability: {probs[idx]:.3f}  (Normal={probs[0]:.3f}, Abnormal={probs[1]:.3f})"
                )
        except Exception as e:
            messagebox.showerror("Error", f"Inference failed:\n{e}")

# ----------------------------
# Entry
# ----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Point this to either:
    #  - a full model file (lung_sound_model_full.pth), or
    #  - a state dict file (lung_sound_model.pth)
    ckpt_path = "lung_sound_model_full.pth"

    if USE_DND:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()

    App(root, device, ckpt_path=ckpt_path)
    root.mainloop()

if __name__ == "__main__":
    main()
