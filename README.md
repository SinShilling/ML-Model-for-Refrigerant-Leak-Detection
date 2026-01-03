# ML-Model-for-Refrigerant-Leak-Detection

## Overview
This project implements a **Python-based machine learning regression model** to detect abnormal behavior in system temperature data.  
The solution uses **Ridge Regression** to predict expected values under normal conditions and identifies anomalies based on **prediction error analysis**.

The primary focus of this project is **data preprocessing, regression modeling, and anomaly detection using Python**.

---

## Problem Statement
Late detection of abnormal system behavior can lead to performance degradation and equipment damage.  
Traditional threshold-based monitoring is often unreliable under varying operating conditions.

The objective of this project is to:
- Learn normal data patterns using regression
- Predict expected values
- Detect anomalies when deviations exceed a defined error threshold

---

## Dataset
The dataset consists of structured numerical features derived from system and environmental measurements.

### Input Features
- Discharge Temperature  
- Ambient Temperature  
- Ambient Humidity  

### Target Variable
- Suction Temperature (healthy condition)

The dataset is preprocessed using **feature scaling** to ensure stable and reliable model performance.

---

## Approach
1. Data cleaning and preprocessing using Python  
2. Feature scaling using standard normalization techniques  
3. Model training using **Ridge Regression** to handle multicollinearity  
4. Prediction of expected values under normal conditions  
5. Anomaly detection using prediction error thresholding  

---

## Model Used
**Ridge Regression (L2 Regularization)**

Ridge Regression was chosen to:
- Reduce overfitting
- Handle correlated input features
- Improve model stability on noisy data

---

## Results
- Accurate prediction of target values under normal operating conditions  
- Clear separation between normal and anomalous behavior using error analysis  
- Consistent and stable model performance after regularization  

---

## Technologies Used
- Python  
- NumPy  
- Pandas  
- Scikit-learn  
- Matplotlib  

---

## How to Run
```bash
pip install -r requirements.txt
python model_training.py
