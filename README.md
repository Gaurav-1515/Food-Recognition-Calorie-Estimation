# Food-Recognition-Calorie-Estimation
## Project Overview

**Comillas Negras — Monoline Solution Icon**

This repository implements a deep learning pipeline to **recognize food items from images** and **estimate their calorie content**. The goal is to enable users to track meals automatically and make informed dietary choices.

Key features:

* Food classification over common dishes (trained on Food-101 dataset or custom dataset).
* Per-image calorie estimation using a combination of class-based average calories and visual/size heuristics.
* End-to-end training pipeline, evaluation scripts, and inference API.

---

## Table of Contents

1. Project summary
2. Dataset
3. Model architecture
4. Calorie estimation approach
5. Installation
6. Usage
7. Training
8. Evaluation
9. Results & expected output
10. File structure
11. Troubleshooting
12. License & citation

---

## 1. Project summary

This project performs two tasks:

* **Classification:** Given an input image, predict the food class.
* **Calorie estimation:** Given the predicted class (and optional portion size), estimate calorie content.

We balance accuracy and practicality by combining a classifier (CNN / transfer-learning) with a lightweight calorie estimator that uses class-average calorie values adjusted by visual cues when available.

---

## 2. Dataset

Primary dataset used: **Food-101** ([https://www.kaggle.com/dansbecker/food-101](https://www.kaggle.com/dansbecker/food-101)).

You may also use a custom dataset. Expected format (for Food-101 style):

```
/dataset
  /images
    /class_1
      img1.jpg
      img2.jpg
    /class_2
      ...
  train.txt
  test.txt
```

For calorie mapping, a CSV (`calorie_lookup.csv`) stores per-class average calories per typical serving, e.g.:

```
class_name,calories_per_serving
pizza,285
burger,354
sushi,200
...
```

---

## 3. Model architecture

We recommend starting with a transfer-learning backbone for reliable results with limited compute:

* Backbone: `EfficientNetB0` or `ResNet50` pretrained on ImageNet.
* Head: GlobalAveragePooling -> Dense(512, ReLU) -> Dropout(0.4) -> Dense(num_classes, softmax).

Example hyperparameters used in experiments:

* Optimizer: Adam (lr=1e-4)
* Batch size: 32
* Image size: 224×224 (adjustable)
* Augmentation: random flip, rotation, brightness jitter, random crop

---

## 4. Calorie estimation approach

Three-tier approach (progressive accuracy):

1. **Class-average lookup (baseline)**

   * Use `calorie_lookup.csv` to return average calories for predicted class.
   * If user provides portion size (e.g., 1.5 servings), multiply accordingly.

2. **Visual heuristics (improved)**

   * Estimate portion area fraction using simple segmentation (color/threshold) or bounding-box size relative to image.
   * Adjust calories by area ratio compared to a reference-serving area.

3. **Scale-aware estimation (advanced)**

   * If the image contains a reference object (e.g., coin, card), compute scale and refine volume estimate.
   * Optional: depth or multiple-angle images for volume -> calories mapping.

Notes: This project implements baseline + visual heuristics; scale-aware estimation is left as an advanced extension.

---

## 5. Installation

Prerequisites (tested with Python 3.10+):

* pip
* virtualenv (recommended)

Install dependencies:

```bash
python -m venv venv
source venv/bin/activate    # macOS / Linux
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

Example `requirements.txt` (suggested):

```
tensorflow>=2.12
numpy
pandas
opencv-python
scikit-learn
matplotlib
tqdm
albumentations
```

---

## 6. Usage

### Prepare dataset

1. Download Food-101 and place images in `dataset/images/`.
2. Ensure `calorie_lookup.csv` exists in `data/`.

### Train model (example)

```bash
python train.py --data_dir dataset --epochs 25 --batch_size 32 --img_size 224 --backbone efficientnetb0
```

### Evaluate model

```bash
python evaluate.py --model checkpoints/best_model.keras --data_dir dataset --batch_size 32
```

### Inference (single image)

```bash
python predict.py --model checkpoints/best_model.keras --image samples/pizza.jpg --output results.json
```

`predict.py` returns a JSON with fields:

```json
{
  "predicted_class": "pizza",
  "probability": 0.89,
  "estimated_calories": 342.0,
  "servings": 1.2
}
```

---

## 7. Training details

* Data augmentation is done using `albumentations` for better robustness.
* Early stopping on validation loss with patience of 6 epochs.
* Use class weights if your dataset is imbalanced.
* Save model in native Keras format e.g. `model.save('best_model.keras')`.

Tips for quicker experiments:

* Freeze backbone for first few epochs, then unfreeze and fine-tune with a smaller learning rate.

---

## 8. Evaluation

Recommended metrics:

* Top-1 accuracy, Top-5 accuracy (if many classes)
* Precision/Recall/F1 per-class
* Confusion matrix

For calorie estimation, measure mean absolute error (MAE) and mean absolute percentage error (MAPE) against ground-truth calories if available.

---

## 9. Results & expected output

Include an `experiments/` folder to store sample evaluation outputs, plots, and confusion matrices. A typical inference JSON output was shown in Section 6.

---

## 10. File structure

```
Task-05/
├── data/
│   └── calorie_lookup.csv
├── dataset/
│   └── images/...
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
├── checkpoints/
├── samples/
├── requirements.txt
└── README.md
```

---

## 11. Troubleshooting

* **ModuleNotFoundError for TensorFlow**: make sure virtualenv is activated and `pip install -r requirements.txt` completed successfully.
* **Shape mismatch on model.load**: verify image size used in training and inference are the same.
* **Poor calorie estimates**: improve `calorie_lookup.csv` with better per-class serving info or collect ground-truth portion sizes.

---

## 12. License & citation

This repository is released under the **MIT License**. Please cite the project as:

> Task-05: Food Recognition & Calorie Estimation — Comillas Negras. (2025). GitHub repository.

---

## 13. Future work / TODO

* Implement scale-aware calibration using reference objects.
* Add multi-task learning: simultaneously predict class and portion size.
* Mobile/edge optimization (TFLite model export).
* Collect a small labeled dataset with per-image calorie ground truth for better calibration.

---

## Contact

For questions or contributions, open an issue or contact the maintainer.

---

*Generated README — adapt any section to your needs.*
