# 🌿 Plant Disease Detection using CNN & Transfer Learning

> Comparing a **Custom CNN** vs **ResNet50 Transfer Learning** for tomato disease classification using the PlantVillage dataset.

**Program:** Deep Neural Networks  

---

## 📊 Results

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|----------|-----------|--------|----------|---------------|
| Custom CNN (from scratch) | 93.9% | 94.1% | 93.2% | 93.5% | ~192s |
| ResNet50 (Transfer Learning) | 97.5% | 97.6% | 97.5% | 97.5% | ~106s |

> **Primary metric: Recall** — In plant disease detection, missing a diseased leaf (false negative) causes undetected disease spread, which is far more costly than a false alarm on a healthy plant.

---

## 🍅 Problem Statement

Early detection of plant diseases can prevent significant crop loss for farmers. This project builds an image classifier to identify 5 types of tomato leaf conditions:

| Class | Type |
|-------|------|
| Tomato Bacterial Spot | Disease |
| Tomato Early Blight | Disease |
| Tomato Late Blight | Disease |
| Tomato Leaf Mold | Disease |
| Tomato Healthy | Healthy |

---

## 🏗️ Model Architectures

### Custom CNN
```
Input (128×128×3)
    ↓
Conv2D(64) → Conv2D(64) → MaxPool → BatchNorm → Dropout(0.25)
    ↓
Conv2D(128) → Conv2D(128) → MaxPool → BatchNorm → Dropout(0.30)
    ↓
Conv2D(256) → MaxPool → BatchNorm → Dropout(0.30)
    ↓
GlobalAveragePooling2D          ← replaces Flatten+Dense
    ↓
Dropout(0.40) → Dense(5, softmax)
```
- **~2.1M total parameters**
- **Loss reduction: 78.9%** over 10 epochs

### ResNet50 Transfer Learning
```
Input (128×128×3)
    ↓
ResNet50 base (ImageNet weights, FROZEN — 175 layers)
    ↓
GlobalAveragePooling2D          ← replaces Flatten+Dense
    ↓
Dense(512, relu) → Dropout(0.40) → Dense(5, softmax)
```
- **~24M total parameters, only ~1M trainable**
- **Loss reduction: 90.7%** over 10 epochs

---

## 🔑 Key Design Decisions

### Why Global Average Pooling (not Flatten+Dense)?
With only ~5,500 training images, a `Flatten → Dense` head creates thousands of extra weights that the model memorizes rather than learns from. GAP collapses each feature map to a single average value — drastically cutting parameters and acting as a structural regularizer.

### Why freeze ResNet50's base layers?
ImageNet pre-training already encodes universal visual features (edges, textures, color gradients) that transfer well to plant leaf imagery. With limited data, allowing all 24M parameters to update risks destroying those features (catastrophic forgetting) and severe overfitting.

### Why Recall over Accuracy?
The dataset is fairly balanced, so accuracy would work — but in an agricultural context, a false negative (missing a diseased plant) has real-world consequences. Recall directly minimizes false negatives.

---

## 📁 Dataset

**PlantVillage** — loaded via `tensorflow_datasets` (no manual download needed)

| Property | Value |
|----------|-------|
| Source | Hughes & Salathé, 2015 |
| Total images | ~11,000 (5 tomato classes) |
| Train/Test split | 90% / 10% |
| Image size | 128 × 128 × 3 |
| Augmentation | Horizontal flip, vertical flip, brightness, contrast |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| TensorFlow / Keras | Model building and training |
| tensorflow_datasets | PlantVillage dataset loading |
| ResNet50 (Keras Applications) | Pre-trained feature extractor |
| scikit-learn | Metrics (accuracy, precision, recall, F1) |
| NumPy | Array operations |
| Matplotlib / Seaborn | Training curves, confusion matrices |
| Pandas | Comparison table |

---

## 📂 Repository Structure

```
plant-disease-cnn-classification/
│
├── Plant_Disease_Detection_using_CNN_ and_Transfer Learning.ipynb   # Main notebook (all outputs included)
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
└── assets/                           # Screenshots from training runs
    ├── custom_cnn_curves.png
    ├── transfer_learning_curves.png
    └── confusion_matrices.png
```

---

## 👤 Author

**Bhrugu Vyas**  
Senior Data Analyst - JP Morgan Chase

MTech AI/ML — BITS Pilani WILP  

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/bhruguvyas/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/bhruguvyas)
