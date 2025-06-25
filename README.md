# MNIST Digit Classifier (2 vs. Not 2)
This project implements a binary classifier to identify the digit "2" within the MNIST dataset. It utilizes a Stochastic Gradient Descent (SGD) classifier and provides a comprehensive evaluation of the model's performance, including learning curves, precision-recall curves, ROC curves, confusion matrices, and classification reports.

<p align="center">
	<img src="assets/mnist.png" alt="mnist" style="width:100%; max-width:800px;" />
</p>
---

## Project Overview

The goal of this project is to train a model that can accurately distinguish between the digit "2" and all other digits in the MNIST dataset. This is achieved by transforming the multi-class MNIST problem into a binary classification task.

<p align="center">
	<img src="assets/ClassifiedDigit.png" alt="ClassifiedDigit" style="width:100%; max-width:800px;" />
</p>
## Dataset
The MNIST dataset is used, which consists of 70,000 grayscale images of handwritten digits (0-9). The dataset is loaded using scikit-learn's `fetch_openml` function.

<p align="center">
	<img src="assets/Dataset.png" alt="Dataset" style="width:100%; max-width:800px;" />
</p>


## Dependencies

* scikit-learn
* NumPy
* Matplotlib
* Seaborn



## Model Training and Evaluation

The script performs the following steps:

1.  Data Loading and Preprocessing:Loads the MNIST dataset, visualizes a sample digit, splits the data into training and test sets, and creates binary labels (2 vs. not 2).
2.  Feature Scaling: Applies `StandardScaler` to scale the pixel values.
3.  Model Training: Trains an `SGDClassifier` on the scaled training data.
4.  Model Evaluation:
      * Calculates cross-validation accuracy.
      * Calculates test set accuracy.
      * Generates and plots the learning curve.
      * Calculates and prints precision, recall, and F1-score.
      * Generates and plots the precision-recall curve (with AUC).
      * Generates and plots the ROC curve (with AUC).
      * Generates and plots the confusion matrix.
      * Prints the classification report.

## Results

### Learning Curve
The learning curve indicates that the model is performing well, with both training and cross-validation scores converging. There is a slight gap, suggesting potential for improvement with more data or regularization.

<p align="center">
	<img src="assets/ModelLearningCurve.png" alt="ModelLearningCurve" style="width:100%; max-width:800px;" />
</p>

### Classification Report
The classification report shows excellent performance, particularly for the "Not 2" class. The "Is 2" class shows good precision but a lower recall, suggesting room for improvement.
<p align="center">
	<img src="assets/ClassReport.png" alt="ClassReport" style="width:100%; max-width:800px;" />
</p>


### Confusion Matrix

The confusion matrix shows a high true negative rate and a moderate false negative rate, consistent with the classification report.

<p align="center">
	<img src="assets/CFmatrix.png" alt="CFmatrix" style="width:100%; max-width:800px;" />
</p>

### Precision-Recall Curve

The precision-recall curve shows an AUC of 0.89, indicating a good trade-off between precision and recall.

<p align="center">
	<img src="assets/PRcurve.png" alt="PRcurve" style="width:100%; max-width:800px;" />
</p>

### ROC Curve
The ROC curve shows an AUC of 0.97, indicating excellent discrimination ability.
<p align="center">
	<img src="assets/ROCcurve.png" alt="ROCcurve" style="width:100%; max-width:800px;" />
</p>
## Conclusion

The model demonstrates strong performance in identifying the digit "2" within the MNIST dataset. Further improvements can be explored through hyperparameter tuning, addressing class imbalance, or gathering more training data.

