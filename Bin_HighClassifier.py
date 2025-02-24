# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 10:07:26 2025

@author: HP
"""

from sklearn.datasets import fetch_openml
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, roc_curve
from sklearn.metrics import f1_score, precision_recall_curve, auc
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler  

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data, mnist.target

# Visualize a digit
some_digit = X.iloc[5]
some_digit_image = some_digit.to_numpy().reshape(28, 28)
plt.imshow(some_digit_image, cmap=mpl.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()

# Convert labels to uint8
y = y.astype(np.uint8)

# Split data into training and test sets
X_train, X_test, y_train, y_test = (X[:60000], X[60000:],
                                    y[:60000], y[60000:])

# Create binary labels (2 vs. not 2)
y_train_2 = (y_train == 2)
y_test_2 = (y_test == 2)

# Feature Scaling (StandardScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training
clf = SGDClassifier(random_state=42)
clf.fit(X_train_scaled, y_train_2)  # Use scaled data

# Predictions
#predict first scaled training image
some_digit_prediction = clf.predict([X_train_scaled[0]]) 

print(f"Prediction for first digit: {some_digit_prediction}")

# Model Evaluations on the training set
model_perf = cross_val_score(clf, X_train_scaled, y_train_2,
                             cv=3, scoring="accuracy")
print(f"Cross-validation accuracy: {model_perf}")

# Model performance on the test set
y_pred = clf.predict(X_test_scaled) #use scaled data
print(f"Predictions on test set: {y_pred}")

test_perf = accuracy_score(y_test_2, y_pred)
print(f"Test set accuracy: {test_perf}")

# Learning Curve
train_sizes, train_scores, test_scores = learning_curve(
    clf, X_train_scaled, y_train_2, cv=5, scoring="accuracy",
    train_sizes=np.linspace(0.1, 1.0, 10) #use scaled data
)

# Calculate mean and standard deviation of scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot the learning curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label="Training score", 
         color="blue")
plt.plot(train_sizes, test_mean, label="Cross-validation score", 
         color="red")
plt.title("Learning Curve")
plt.xlabel("Training set size")
plt.ylabel("Accuracy score")
plt.legend(loc="best")
plt.grid(True)
plt.show()

# Precision, Recall, F1-score
print(f"Precision: {precision_score(y_test_2, y_pred)}")
print(f"Recall: {recall_score(y_test_2, y_pred)}")
print(f"F1-score: {f1_score(y_test_2, y_pred)}")

# Precision-Recall Curve
y_scores = clf.decision_function(X_test_scaled) #use scaled data
precision, recall, thresholds = precision_recall_curve(y_test_2, y_scores)
auc_score = auc(recall, precision)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f"AUC = {auc_score:.2f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend(loc="best")
plt.grid(True)
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test_2, y_scores)
auc_score_roc = roc_auc_score(y_test_2, y_scores)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc_score_roc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, classes, 
                          title='Confusion Matrix', cmap=plt.cm.Blues):
    """Plots a confusion matrix using matplotlib and seaborn."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, 
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show() #added plt.show()

classes = ['Not 2', 'Is 2']
plot_confusion_matrix(y_test_2, y_pred, classes)

# Classification Report
print("Classification Report:")
print(classification_report(y_test_2, y_pred))