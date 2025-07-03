# SVM Classification with Linear and RBF Kernels on Breast Cancer Dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix

# 1.Load and prepare a dataset for binary classification.
#a1. Load and Prepare Dataset
 
# Load your dataset
df = pd.read_csv("breast-cancer.csv")  # Replace with your actual file name

# Example: Suppose your dataset has columns: 'feature1', 'feature2', ..., 'target'
X = df.drop('target', axis=1)  # All columns except target
y = df['target']  # Binary target column

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2.Train an SVM with linear and RBF kernel.

# Reduce dimensions to 2D for visualization
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)

# Split the 2D data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X_2d, y, test_size=0.3, random_state=42)

#  Train SVM Models
svm_linear = SVC(kernel='linear', C=1)
svm_rbf = SVC(kernel='rbf', gamma='scale', C=1)

svm_linear.fit(X_train, y_train)
svm_rbf.fit(X_train, y_train)

# . Visualization function
def plot_decision_boundary(model, X, y, title):
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
    plt.title(title)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.show()

#  3.Visualize decision boundary using 2D data.
plot_decision_boundary(svm_linear, X_train, y_train, "SVM Linear Kernel Decision Boundary")
plot_decision_boundary(svm_rbf, X_train, y_train, "SVM RBF Kernel Decision Boundary")

# 4. Hyperparameter Tuning4.Tune hyperparameters like C and gamma.
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.01, 0.1, 1],
    'kernel': ['rbf']
}

grid = GridSearchCV(SVC(), param_grid, refit=True, cv=5)
grid.fit(X_train, y_train)

print("Best Parameters from Grid Search:", grid.best_params_)

# Evaluate tuned model
y_pred = grid.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 5. Cross-Validation
scores = cross_val_score(grid.best_estimator_, X_scaled[:, :2], y, cv=5)
print("\nCross-validation Accuracy Scores:", scores)
print("Mean Accuracy:", scores.mean())
