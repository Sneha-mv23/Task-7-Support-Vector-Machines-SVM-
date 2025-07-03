# Task-7-Support-Vector-Machines-SVM-



# SVM Classification with Linear and RBF Kernels

## Objective
This project demonstrates the application of Support Vector Machines (SVM) using both linear and RBF kernels for binary classification using the Breast Cancer dataset from Scikit-learn.

## Steps Covered
1. Loaded Breast Cancer dataset and preprocessed features.
2. Reduced feature space to 2D using PCA for visualization.
3. Trained SVM models with linear and RBF kernels.
4. Visualized decision boundaries.
5. Tuned hyperparameters using GridSearchCV.
6. Evaluated model using cross-validation and performance metrics.

## Requirements
- Python 3.x
- NumPy
- pandas
- matplotlib
- scikit-learn

## To Run:
```bash
python svm_classification.py

##  svm-linear_kernel_decision_boundary
 svm-linear_kernel_decision_boundary.png




### output
Best Parameters from Grid Search: {'C': 1, 'gamma': 'scale', 'kernel': 'rbf'}

Classification Report:
               precision    recall  f1-score   support

           0       0.95      0.87      0.91        63
           1       0.93      0.97      0.95       108

    accuracy                           0.94       171
   macro avg       0.94      0.92      0.93       171        
weighted avg       0.94      0.94      0.94       171        

Confusion Matrix:
 [[ 55   8]
 [  3 105]]

Cross-validation Accuracy Scores: [0.85087719 0.87719298 0.9122807  0.93859649 0.90265487]
Mean Accuracy: 0.8963204471355379