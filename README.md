# 🛡️ Secure Face Recognition & Liveness Detection System

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=flat&logo=opencv&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)

## 📌 Overview
This repository contains the implementation of a robust **Facial Recognition and Authentication System** designed to provide secure identity verification and replace traditional physical badge systems for access control. 

The system relies on biometric data acquired via smartphones and is built to be highly robust against diverse environmental conditions. It features an end-to-end pipeline that handles everything from real-time image quality assessment to deep learning-based anti-spoofing (Liveness Detection) and highly accurate identity matching.

## ✨ Key Features
- **🛡️ Liveness Detection (Anti-Spoofing)**: Deep learning-based module to identify replay attacks and presentation attacks, ensuring the physical presence of the user and preventing fraud.
- **👤 User Enrollment**: Secure registration of new users by extracting and storing only biometric features (embeddings) rather than raw image data in the Gallery.
- **🔍 Open-set User Identification**: Identifies a user from a gallery of known individuals, with a built-in *Reject option* for unknown/unregistered users.
- **✅ User Verification**: 1:1 matching to verify a claimed identity (without a Reject option).
- **📸 Quality-Aware Acquisition**: Automatically discards low-quality samples and requests re-acquisition to maintain high matching accuracy.

## 🏗️ System Architecture
The pipeline is divided into several core modules:
1. **Acquisition Module** (acquisition.py): Validates the quality of the acquired image before passing it to the downstream tasks.
2. **Feature Extractor** (feature_extractor.py): Extracts deep embeddings from the biometric samples. Used during both enrollment and matching phases.
3. **Liveness Detector** (liveness_detector.py): Evaluates whether the sample is a genuine presentation or a spoofing attack.
4. **Matching Systems** (identification_system.py, verification_system.py): Computes similarity scores and thresholds for 1:N identification and 1:1 verification.

## 📂 Repository Structure

    📁 docs/                   # Project reports and documentation
    📁 data/                   # Metadata, extracted embeddings (.npy), and dataset splits
    📁 models/                 # Pre-trained weights (e.g., best_weights_spoof.pth)
    📁 notebooks/              # Jupyter Notebooks for training and evaluation
      📁 train/                # Data processing and liveness detector training
      📁 evaluation/           # System evaluation (threshold tuning, testing)
    📁 src/                    # Core source code
      📁 utility/              # Helper scripts (trainer, checkpoint manager)
      📄 biometric_system.py   # Main wrapper for the full authentication pipeline
      📄 config.py             # Global configurations and hyperparameters
      📄 ...                   # Other system modules (acquisition, liveness, etc.)
    📄 requirements.txt        # Python dependencies
    📄 README.md               # This documentation file

## 🚀 Getting Started

### 1. Clone the repository

    git clone https://github.com/YourUsername/Secure-Face-Recognition-and-Liveness-Detection.git
    cd Secure-Face-Recognition-and-Liveness-Detection

### 2. Install dependencies
It is recommended to use a virtual environment (e.g., venv or conda).

    pip install -r requirements.txt

### 3. Setup Data & Models
Place your datasets and pre-extracted features inside the data/ folder.
Ensure the pre-trained weights for the Liveness Detector are placed in the models/ directory.

### 4. Running the Code
You can explore the system step-by-step using the provided Jupyter Notebooks in the notebooks/evaluation/ directory:
- final_system.ipynb: End-to-end demonstration of the full authentication pipeline.
- acquisition_test.ipynb: Tests the quality assessment module.
- identification_threshold.ipynb / verification_threshold.ipynb: Notebooks for tuning the decision thresholds.

To train the liveness detector from scratch, refer to the notebooks in notebooks/train/.

## 👥 Authors
* **Simone Faraulo & Ivan Cipriano**

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.