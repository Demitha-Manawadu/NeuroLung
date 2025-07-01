# 🫁 NeuroLung: AI-Powered Lung Sound Diagnostic Web Application

**NeuroLung** is a low-cost, AI-based system for classifying lung sounds using deep learning and signal processing techniques. It enables early detection of respiratory abnormalities through a browser-accessible Flutter web app, making it suitable for use in rural clinics, medical education, and telemedicine.

---

## 🎯 Objective

To develop a portable, accessible, and intelligent lung sound diagnostic tool that can:

- Analyze auscultated audio for early signs of diseases such as **COPD, asthma, pneumonia, and bronchitis**
- Provide **real-time predictions** via a **Flutter web interface**
- Serve as a tool for **clinical support**, **self-assessment**, and **medical training**

---

## 🧠 System Architecture

### 1. **Audio Acquisition**
- Modified analog or digital stethoscope with a microphone
- Audio interface connected to a laptop or mobile device

### 2. **Preprocessing & Feature Extraction**
- **Denoising** and segmentation
- **STFT → Mel Spectrogram → Log-Mel Spectrogram**
- Feature embeddings via **pretrained VGGish**

### 3. **Deep Learning Pipeline**
- CNN-based classifier for detecting normal vs abnormal sounds
- Multi-class classification (normal, wheezes, crackles, rhonchi)

### 4. **Flutter Web Interface**
- Built using **Flutter Web** for high-performance, cross-platform compatibility
- Communicates with a **Flask-based API backend**
- Displays diagnostic results, confidence levels, and spectrogram visualizations

---

## 🧪 Datasets

### ✔ ICBHI 2017 Challenge Dataset  
- 5.5 hours of respiratory recordings  
- Labels: Normal, Wheeze, Crackle, Both

### ✔ PhysioNet Respiratory Sound Database  
- Extensive pediatric and adult samples  
- Environmental diversity and noise

**Preprocessing Techniques:**
- Normalization, silence trimming, pitch/time augmentation, filtering

---

## 🧬 Machine Learning Model

| Component       | Details                          |
|----------------|----------------------------------|
| Input           | Log-Mel Spectrograms (Image-like) |
| Feature Extractor | VGGish (transfer learning)       |
| Classifier      | Custom CNN                       |
| Loss Function   | Categorical Crossentropy         |
| Optimizer       | Adam                             |
| Metrics         | Accuracy, Sensitivity, AUC       |

---

## 💻 Technologies Used

| Layer             | Technology                       |
|------------------|----------------------------------|
| Frontend          | Flutter Web (Dart)               |
| Backend API       | Python (Flask)                   |
| ML Frameworks     | TensorFlow, Keras, PyTorch       |
| Audio Processing  | Librosa, NumPy, SciPy            |
| Visualization     | Flutter Charts, Matplotlib       |
| Data Storage      | (Optional) MySQL/PostgreSQL      |

---

## 🧩 Key Features

- 🎙️ **Real-time Lung Sound Classification**
- 🌐 **Cross-platform Web App (Flutter)**
- 📊 **Spectrogram and Prediction Visualization**
- 🏥 **Remote-friendly and lightweight**
- 🩺 **Designed for students, doctors, and patients**

---

## 🧭 Application Scope

| Use Case         | Description |
|------------------|-------------|
| **Medical Training** | Interactive auscultation learning with real sound patterns |
| **Pre-diagnostic Tool** | Assist GPs or self-assessment before clinical visits |
| **Rural Healthcare** | Used where expert pulmonologists aren’t available |
| **Telehealth** | Enables remote screening and referral support |

---

## 📈 Future Enhancements

- Integration of mobile Flutter app (Android/iOS)
- Model explainability via Grad-CAM or SHAP
- Bluetooth-based wireless stethoscope support
- Clinical validation and regulatory preparation
- User login and diagnostic history dashboard

---

## 👥 Project Contributors

Developed by undergraduate researchers from the **University of Moratuwa** and **Faculty of Medicine, University of Colombo**  
**Mentor:** M. Demitha Manawadu

| Name                          | Role                           |
|-------------------------------|--------------------------------|
| L.G. Chamudi Ransika          | DSP & Audio Pipeline           |
| V.Gihansa Gunesekara          | Flutter App & Architecture     |
| Shasmitha Akashwara Liyanage  | Clinical Advisory & Validation |
| S. Dilshan Wickramasinghe     | AI Model Development           |

---


