# Terrain Detection via Satellite Image Segmentation

> MSc Dissertation Project — University of Strathclyde 

A deep learning pipeline for semantic segmentation of terrain features from satellite imagery, developed for autonomous vehicle and remote-sensing applications.

---

## Problem Statement

Autonomous vehicles and remote-sensing systems need to reliably identify terrain types — roads, vegetation, water, bare ground — from overhead imagery. Manual labelling is not scalable. This project explores whether a deep learning segmentation model can accurately classify terrain features at pixel level from satellite images, even under challenging lighting and background conditions.

---

## Approach

```
Raw Satellite Images
        │
        ▼
Data Preparation
(cleaning, mask validation, resizing, augmentation)
        │
        ▼
Model Training
(U-Net / EfficientNet encoder-decoder)
        │
        ▼
Evaluation
(IoU, Dice coefficient, visual inspection)
        │
        ▼
Tuning and Analysis
(augmentation strategy, class balance, learning rate)
```

---

## Key Results

| Metric | Score |
|--------|-------|
| IoU (Intersection over Union) | Reported in notebook |
| Dice Coefficient | Reported in notebook |
| Model Architecture | U-Net with EfficientNet encoder |

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

- Python, PyTorch / TensorFlow
- U-Net architecture with EfficientNet encoder
- Data augmentation (flipping, rotation, colour jitter, lighting variation)
- IoU and Dice coefficient evaluation
- Jupyter Notebook

---

## Project Structure

```
Terrain-Detection/
│
├── notebooks/
│   └── terrain_segmentation.ipynb   # Main pipeline notebook
│
├── data/
│   ├── images/                      # Satellite image samples
│   └── masks/                       # Ground truth segmentation masks
│
├── models/
│   └── model_weights/               # Saved model checkpoints
│
├── outputs/
│   └── predictions/                 # Example segmentation outputs
│
├── requirements.txt
└── README.md
```

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/zainhammagi12/Terrain-Detection-using-Satellite-Data-for-Autonomous-Vehicle

# Install dependencies
pip install -r requirements.txt

# Open the notebook
jupyter notebook notebooks/terrain_segmentation.ipynb
```

---

## Methodology Detail

**Data Preparation**
Images and masks were cleaned to remove corrupted files, resized to a consistent input dimension, and split into training, validation, and test sets. Augmentation included horizontal and vertical flipping, rotation, brightness and contrast variation, and synthetic shadow overlays to improve generalisation across environments.

**Model Architecture**
An encoder-decoder architecture was used with a pre-trained EfficientNet backbone as the encoder (transfer learning) and a U-Net style decoder with skip connections to recover spatial resolution. This combination balances feature extraction depth with segmentation precision.

**Training**
Trained with a combined loss function (cross-entropy and Dice loss) to handle class imbalance between terrain categories. Learning rate scheduling was applied to improve convergence.

**Evaluation**
Primary metrics were IoU (Intersection over Union) per class and mean Dice coefficient. Visual inspection of prediction masks against ground truth was used to identify systematic errors in challenging scenes.

---

## Limitations and Future Work

- Dataset size limits generalisation to novel geographies not represented in training
- Real-time inference optimisation not yet implemented
- Multi-temporal imagery (seasonal variation) could improve robustness
- Deployment as an API endpoint would extend usability for downstream systems

---

## Author

**Zain Hammagi** — [linkedin.com/in/zain-hammagi](https://linkedin.com/in/zain-hammagi) · [zainhammagi.github.io](https://zainhammagi.github.io)
