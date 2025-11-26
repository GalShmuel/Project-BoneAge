# Bone Age Prediction from X-Ray Images

A deep learning project for predicting bone age from hand X-ray images using multiple neural network architectures. This project implements and compares three different models: ResNet50, EfficientNetB0, and a custom CNN architecture.

## 📋 Project Overview

This project predicts bone age (in years) from pediatric hand X-ray images. Bone age is a measure of skeletal maturity commonly used in pediatric medicine. The models were trained, validated, and tested on a labeled X-ray dataset.

### Key Features

* **Multiple Model Architectures**: ResNet50, EfficientNetB0, and custom CNN
* **Data Augmentation**: Improves model generalization
* **Comprehensive Evaluation**: MAE, RMSE, R² metrics
* **Transfer Learning**: Uses pre-trained ImageNet weights
* **Visualization Tools**: Functions to visualize predictions and training history

## 💾 Project Structure

```
Project-BoneAge/
│
├── Data/
│   ├── TrainingSet/
│   │   ├── Images/
│   │   └── labels.csv
│   ├── ValidationSet/
│   │   ├── Images/
│   │   └── labels.csv
│   └── TestSet/
│       ├── Images/
│       └── labels.csv
│
├── DeepLearningProject.ipynb
├── README.md
```

## 💾 Data Folder Note

> **Important:** The `Data/` folder containing the X-ray images (~10 GB) is **not included** in this repository due to GitHub size limitations.
> To run this project, you need to provide your own dataset or use placeholders.
>
> The folder structure should look like this:
>
> ```
> Project-BoneAge/Data/
> ├── TrainingSet/Images/
> ├── TrainingSet/labels.csv
> ├── ValidationSet/Images/
> ├── ValidationSet/labels.csv
> └── TestSet/Images/
>     └── labels.csv
> ```
>
> You can create sample images and CSV files for testing purposes if the full dataset is unavailable.

## 🚀 Usage

### Running the Notebook

1. Open the Jupyter Notebook:

```bash
jupyter notebook DeepLearningProject.ipynb
```

2. Execute cells sequentially:

   * Data preparation → Model creation → Training → Evaluation

### Training Models

#### ResNet50 Training

1. Initial training on the training set
2. Fine-tuning on validation set

#### EfficientNetB0 Training

* Run notebook cells 26–27

#### Custom CNN Training

* Run notebook cells 35–43

## 📊 Results

**ResNet50 Model** (best performance)

* MAE: 1.03 years
* RMSE: 1.29 years
* R² Score: 0.8697

**EfficientNetB0 Model**

* MAE: 1.37 years
* RMSE: 1.73 years
* R² Score: 0.7297

**Custom CNN Model**

* MAE: 1.07 years
* RMSE: 1.32 years
* R² Score: 0.7746

## 🔧 Key Components

* **Normalization**: Z-score for bone age values
* **Age Binning**: Stratified splits for balanced train/validation
* **Image Preprocessing**: Resize to 224x224, normalize pixel values to [0,1]
* **Data Augmentation**: Flip, rotation, brightness/contrast/saturation/hue adjustments
* **Training Features**: Callbacks (ReduceLROnPlateau, ModelCheckpoint, EarlyStopping), Adam optimizer, L2 regularization
* **Evaluation Metrics**: MAE, RMSE, R²

## 🎯 Future Improvements

* Ensemble multiple models
* Advanced augmentation techniques
* Hyperparameter tuning
* Cross-validation for robust evaluation
* Integration with clinical workflow

## 📜 License

Educational purposes only - part of a deep learning course project.

## 👥 Authors

Created as part of a Deep Learning course project.
