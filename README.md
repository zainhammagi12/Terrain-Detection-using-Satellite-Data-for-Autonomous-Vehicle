# Terrain-Detection-using-Satellite-Data-for-Autonomous-Vehicle

Project Overview
This project develops a comprehensive terrain detection system using satellite imagery to enhance autonomous vehicle navigation. The system combines two deep learning approaches:

Terrain Classification using EfficientNetV2 (97% accuracy)
Road Segmentation using U-Net (0.50 Dice coefficient)

Key Features

Multi-model approach combining classification and segmentation
Processing of 27,000+ satellite images from EuroSAT dataset
Road detection using Massachusetts Roads Dataset (1,171 images)
Comprehensive evaluation metrics and visualization

Technical Implementation
Terrain Classification

Model: EfficientNetV2
Dataset: EuroSAT (27,000 images, 10 classes)
Performance: 97% accuracy
Features:

Data augmentation
Early stopping
RMSprop optimizer
Batch normalization



Road Segmentation

Model: U-Net
Dataset: Massachusetts Roads Dataset
Performance: 0.50 Dice coefficient
Features:

Custom loss function (Soft Dice Loss)
Image cropping for efficient processing
Binary mask generation
Advanced data preprocessing



Results

Terrain Classification:

97% testing accuracy
96.8% F1 score
Robust performance across different terrain types


Road Segmentation:

0.50 Dice coefficient
Effective road network detection
Robust handling of complex urban scenarios



Technologies Used

Python 3.10.12
TensorFlow 2.12.0
OpenCV
NumPy, Pandas
Scikit-learn
Matplotlib, Seaborn

Code Structure
Copyterrain-detection/
├── code/
│   ├── terrain_classification.py      # EfficientNetV2 implementation
│   ├── road_segmentation.py          # U-Net implementation
│   ├── data_preprocessing.py         # Data handling and preprocessing
│   └── utils.py                      # Helper functions and metrics
└── README.md

Setup and Installation
bashCopy# Clone the repository
git clone https://github.com/zainhammagi12/terrain-detection.git

# Install dependencies
pip install -r requirements.txt

# Run the classification model
python code/terrain_classification.py

# Run the segmentation model
python code/road_segmentation.py
Model Performance
Terrain Classification Results

Overall Accuracy: 97%
F1 Score: 96.8%
Robust performance across different terrain types
Effective handling of edge cases

Road Segmentation Results

Dice Coefficient: 0.50
Effective road network detection
Robust handling of complex urban scenarios

Future Improvements

Implementation of real-time processing capabilities
Integration with vehicle navigation systems
Enhanced model optimization for edge devices
Extension to diverse geographical regions

Author
Mohammad Zain Hammagi
MSc Advanced Computer Science
University of Strathclyde
