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
numpy: 2.2.3
pandas: 2.3.1
matplotlib: 3.10.0
scikit-learn: 1.7.2
xgboost: 3.2.0
torch: 2.6.0+cu124
pickle: built-in Python standard library
```

To install the main required packages, run:

```bash
pip install numpy pandas matplotlib scikit-learn xgboost torch
```

## Repository Organization Note

The `notebooks/` folder contains the experimental development process. These notebooks include exploratory trials using standard libraries such as scikit-learn, PyTorch, and XGBoost to compare methods, test preprocessing choices, and analyze model behavior.

The `src/` folder contains the finalized implementation used for the formal version of the project. In particular, the final manual Linear SVM implementation and XG-Boost are placed under `src/`.

This separation is used to distinguish exploratory experiments from the final project implementation.

## Dataset

This project uses the UCI SECOM wafer manufacturing dataset. Each sample contains semiconductor manufacturing sensor measurements, and the label indicates whether the wafer passed or failed.

The label definition is:

```text
-1 = Pass
 1 = Fail
```

The dataset is highly imbalanced. Most samples are Pass samples, while Fail samples only represent a small portion of the dataset. Because of this, accuracy alone is not enough to evaluate the model. In this project, recall for the Fail class is especially important.

## Preprocessing and Feature Engineering

The basic preprocessing pipeline includes the following steps:

1. Drop the `Time` column.
2. Remove features with more than 50% missing values.
3. Fill remaining missing values using median imputation.
4. Remove constant features.
5. Remove highly correlated features.
6. Split the dataset into training, validation, and test sets.
7. Standardize features using training-set statistics.

After the basic preprocessing steps, two additional feature engineering methods were tested:

- PCA-based preprocessing
- VAE-based preprocessing

PCA was used to reduce the feature dimension and create a lower-dimensional representation of the data. The VAE-based method was used to generate reconstruction-error-related features for anomaly detection and classification experiments.

The main preprocessing notebooks are:

```text
notebooks/Preprocess_Check.ipynb
notebooks/Preprocess_PCA.ipynb
notebooks/Preprocess_VAE_pass.ipynb
```

## Models

After preprocessing, several training and detection methods were tested:

- SVM
- XGBoost
- Random Forest
- Outlier detection methods

The outlier detection experiments include methods such as Isolation Forest and One-Class SVM. These methods were tested because the Fail class is rare in the dataset, so Fail samples may behave like abnormal samples compared with Pass samples.

However, the outlier detection methods and Random Forest did not perform well in our experiments. The outlier detection methods had difficulty separating Fail samples from normal Pass samples, and Random Forest tended to predict most samples as the majority Pass class. Because of their weak recall on the Fail class, these methods were not selected as final models.

As a result, the later experiments focused mainly on SVM and XGBoost, which showed better ability to detect Fail samples after class weighting, feature engineering, and threshold tuning.

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


## Notes on Results

The experiments show a clear trade-off between recall and precision. Increasing recall helps detect more Fail samples, but it often increases the number of false positives. This is a major challenge because the dataset is highly imbalanced.

The manual Linear SVM was useful for understanding the core SVM training logic, but its performance was limited compared with optimized library implementations. This is expected because standard machine learning libraries use more advanced and stable optimization solvers.


## How to Run

A typical workflow is:

1. Run the preprocessing notebooks in the `notebooks/` folder.
2. The preprocessing notebooks automatically save the processed `.pkl` files into the `data/` folder.
3. Run the model notebooks, such as `SVM.ipynb` or `xg_boost.ipynb` in the `src/` folder.
4. Compare validation and test results using the printed metrics and confusion matrices.

Recommended order:

```text
1. notebooks/Preprocess_Check.ipynb
2. notebooks/Preprocess_PCA.ipynb
3. notebooks/Preprocess_VAE_pass.ipynb
4. notebooks/SVM.ipynb
5. notebooks/xg_boost.ipynb
```
