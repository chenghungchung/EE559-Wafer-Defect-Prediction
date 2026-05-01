# EE559 Wafer Defect Prediction

This repository contains the final project for EE559: Wafer Defect Prediction using the SECOM dataset. The goal of this project is to detect wafer failure samples from high-dimensional semiconductor manufacturing sensor data.

Because the dataset is highly imbalanced, the main focus of this project is improving the recall of the Fail class while also monitoring precision, F1-score, F2-score, accuracy, and the confusion matrix.

## Environment

This project was developed and tested using:

```text
Python 3.11.9
```

Main Python packages used:

```text
numpy
pandas
matplotlib
scikit-learn
xgboost
torch
pickle
```

To install the main required packages, run:

```bash
pip install numpy pandas matplotlib scikit-learn xgboost torch
```

## Project Structure

```text
EE559-Wafer-Defect-Prediction/
│
├── data/
│   ├── uci-secom.csv
│   ├── preprocessed_data.pkl
│   ├── preprocessed_data_outlier.pkl
│   ├── preprocessed_data_outlier_pca.pkl
│   ├── preprocessed_data_outlier_vae.pkl
│   └── other processed data files
│
├── notebooks/
│   ├── Preprocess_Check.ipynb
│   ├── Preprocess_PCA.ipynb
│   ├── Preprocess_VAE.ipynb
│   ├── outlier.ipynb
│   ├── SVM.ipynb
│   └── xg_boost.ipynb
│
├── src/
│   ├── SVM.ipynb
│   └── utils.py
│
├── results/
│
├── README.md
└── .gitignore
```

## Dataset

This project uses the UCI SECOM wafer manufacturing dataset. Each sample contains semiconductor manufacturing sensor measurements, and the label indicates whether the wafer passed or failed.

The label definition is:

```text
-1 = Pass
 1 = Fail
```

The dataset is highly imbalanced. Most samples are Pass samples, while Fail samples only represent a small portion of the dataset. Because of this, accuracy alone is not enough to evaluate the model. In this project, recall for the Fail class is especially important.

## Preprocessing

The preprocessing pipeline includes the following steps:

1. Drop the `Time` column.
2. Remove features with more than 50% missing values.
3. Fill remaining missing values using median imputation.
4. Remove constant features.
5. Remove highly correlated features.
6. Split the dataset into training, validation, and test sets.
7. Standardize features using training-set statistics.
8. Apply additional feature engineering methods such as PCA or VAE-based features.

The main preprocessing notebooks are:

```text
notebooks/Preprocess_Check.ipynb
notebooks/Preprocess_PCA.ipynb
notebooks/Preprocess_VAE.ipynb
```

## Feature Engineering

Several feature representations were tested in this project.

### Original Preprocessed Features

The original preprocessed features were obtained after missing-value handling, constant-feature removal, correlation-based feature removal, and standardization.

### PCA Features

Principal Component Analysis (PCA) was used to reduce the feature dimension. PCA was fitted only on the training set, and the same transformation was applied to the validation and test sets.

### VAE-Based Features

A VAE-based feature engineering approach was also tested. The VAE was trained using Pass samples to learn the normal data pattern. The reconstruction error was then used as an additional anomaly-related feature.

## Models

The following models or experimental settings were tested:

- Linear SVM
- Manual Linear SVM
- PCA + Linear SVM
- Outlier detection methods
- VAE-enhanced features
- XGBoost

For the manual Linear SVM experiment, the SVM training logic, prediction function, confusion matrix, and evaluation metrics were implemented directly instead of using scikit-learn model functions.

## Evaluation Metrics

The models were evaluated using:

```text
Accuracy
Precision
Recall
F1-score
F2-score
Confusion Matrix
```

Since the main goal is wafer failure detection, recall for the Fail class is emphasized. A higher recall means the model detects more actual Fail samples. However, precision and the confusion matrix are also important because a model with very high recall may incorrectly classify many Pass samples as Fail.

## Final Manual SVM Experiment

One final manual Linear SVM experiment used PCA features with the following configuration:

```text
C = 0.05
class_weight = {-1: 1.0, 1: 5.0}
learning_rate = 5e-05
threshold = -0.0655546589142745
```

The final test result was:

```text
Accuracy: 0.5636
Precision: 0.1081
Recall: 0.7500
F1-score: 0.1890
F2-score: 0.3429

Confusion Matrix:
[[121  99]
 [  4  12]]
```

This model detected 12 out of 16 Fail samples on the test set. However, it also produced many false positives, meaning many Pass samples were incorrectly predicted as Fail.

## Notes on Results

The experiments show a clear trade-off between recall and precision. Increasing recall helps detect more Fail samples, but it often increases the number of false positives. This is a major challenge because the dataset is highly imbalanced.

The manual Linear SVM was useful for understanding the core SVM training logic, but its performance was limited compared with optimized library implementations. This is expected because standard machine learning libraries use more advanced and stable optimization solvers.

## Academic Integrity

Standard Python libraries such as NumPy, pandas, scikit-learn, PyTorch, XGBoost, and matplotlib were used as tools for data processing, modeling, and visualization. The preprocessing decisions, feature engineering choices, model comparison, threshold selection, and result analysis were designed and implemented as part of this project.

No external project-specific code was copied directly into this repository.

## How to Run

A typical workflow is:

1. Run the preprocessing notebooks in the `notebooks/` folder.
2. Save the processed `.pkl` files into the `data/` folder.
3. Run the model notebooks, such as `SVM.ipynb` or `xg_boost.ipynb`.
4. Compare validation and test results using the printed metrics and confusion matrices.

Recommended order:

```text
1. notebooks/Preprocess_Check.ipynb
2. notebooks/Preprocess_PCA.ipynb
3. notebooks/SVM.ipynb
4. notebooks/xg_boost.ipynb
```

## Repository Notes

The following files should not be committed:

```text
__pycache__/
*.pyc
.ipynb_checkpoints/
```

These files are automatically generated by Python or Jupyter and are not needed to run the project.