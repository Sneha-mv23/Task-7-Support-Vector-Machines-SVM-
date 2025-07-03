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
 svm_linear_kernel_decision_boundary.png

##  svm-rbf_kernel_decision_boundary
 svm_rbf_kernel_decision_boundary.png




### output
         id diagnosis  radius_mean  texture_mean  perimeter_mean  ...  compactness_worst  concavity_worst  concave points_worst  symmetry_worst  fractal_dimension_worst
0    842302         M        17.99         10.38          122.80  ...             0.6656           0.7119                0.2654          0.4601                  0.11890
1    842517         M        20.57         17.77          132.90  ...             0.1866           0.2416                0.1860          0.2750                  0.08902
2  84300903         M        19.69         21.25          130.00  ...             0.4245           0.4504                0.2430          0.3613                  0.08758
3  84348301         M        11.42         20.38           77.58  ...             0.8663           0.6869                0.2575          0.6638                  0.17300
4  84358402         M        20.29         14.34          135.10  ...             0.2050           0.4000                0.1625          0.2364                  0.07678

[5 rows x 32 columns]
Best Parameters from Grid Search: {'C': 1, 'gamma': 'scale', 'kernel': 'rbf'}

Classification Report:
               precision    recall  f1-score   support

           0       0.93      0.97      0.95       108
           1       0.95      0.87      0.91        63

    accuracy                           0.94       171
   macro avg       0.94      0.92      0.93       171
weighted avg       0.94      0.94      0.94       171

Confusion Matrix:
 [[105   3]
 [  8  55]]

Cross-validation Accuracy Scores: [0.85087719 0.87719298 0.9122807  0.93859649 0.90265487]
Mean Accuracy: 0.8963204471355379