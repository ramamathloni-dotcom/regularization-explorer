# Learning Curve Analysis

This repository contains the analysis of the telecom churn dataset. The objective is to evaluate the model's performance using learning curves to diagnose issues like overfitting.

## Project Structure
- `data/`: Contains the `telecom_churn.csv` dataset.
- `learning_curve_analysis.py`: Python script used to compute scores and generate the learning curve plot.
- `learning_curve_result.png`: The visual output of the learning curve analysis.

## Key Findings
- The learning curve reveals a significant gap between training and validation scores, indicating an **overfitting** issue.
- The model performs perfectly on the training set (score of 1.0) but fails to generalize on new data (validation score around 0.5).
- Suggested improvements include applying regularization (e.g., L2) or feature selection to reduce model complexity.

## How to Run
1. Ensure you have the required libraries installed:
   ```bash
   pip install pandas numpy matplotlib scikit-learn