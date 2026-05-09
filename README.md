# Hotel Booking Cancellation Prediction

This project implements an end-to-end machine learning pipeline to predict whether a hotel booking will be canceled. The objective is to support hotels in revenue management and operational planning by identifying high-risk reservations in advance.

## Problem Overview
Hotel booking cancellations negatively impact revenue, room utilization, and demand forecasting. This project formulates the task as a binary classification problem where the goal is to predict booking cancellation based on historical reservation data.

## Dataset
The dataset contains historical hotel booking records with features related to booking details, customer demographics, stay duration, pricing, and reservation status.

**Target Variable**
- `is_canceled`
  - 1 → booking canceled
  - 0 → booking not canceled

## Data Preprocessing
- Handling missing values (agent, country, children, company)
- Removal of data leakage features
- Duplicate removal and outlier handling
- One-hot encoding of categorical variables
- Feature scaling for distance-based models

## Exploratory Data Analysis (EDA)
Key insights discovered:
- Strong class imbalance between canceled and non-canceled bookings
- Longer lead time significantly increases cancellation probability
- Higher Average Daily Rate (ADR) bookings cancel more frequently
- Seasonal trends strongly affect cancellations

## Class Imbalance Handling
- Applied **SMOTE** on the training dataset only
- Improved recall and F1-score for canceled bookings

## Feature Engineering & Selection
- Created new features such as total stay duration
- Applied **Genetic Algorithm-based feature selection** to reduce dimensionality and improve generalization

## Models Implemented
- K-Nearest Neighbors (KNN)
- Decision Tree
- Neural Network (MLP Classifier)

All models were trained using the same data splits and selected feature set for fair comparison.

## Evaluation
- Accuracy, Precision, Recall, F1-score
- Confusion Matrix analysis

**Best Model:**  
✅ Multi-Layer Perceptron (MLP), achieving the strongest overall performance in recall and F1-score.

## Technologies
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook

## Future Work
- Add external features (holidays, events)
- Explore ensemble models
- Apply explainability techniques (e.g. SHAP)
