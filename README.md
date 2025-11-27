# Handwritten-Number-Detection

**MNIST Digit Classification — SGD Classifier**

This notebook demonstrates building a binary digit classifier (detecting the digit “5”) using the MNIST dataset from OpenML. It covers data loading, visualization, training a linear classifier with SGDClassifier, evaluating with cross-validation, and analyzing results with confusion matrix, precision, and recall.

**Features**

Load the MNIST dataset using fetch_openml.

Visualize sample digits as 28×28 images.

Train a stochastic gradient descent (SGD) linear classifier to detect the digit “5”.

Evaluate model performance using cross-validation, confusion matrix, precision, and recall.

Compare with a baseline dummy classifier.

Save figures and outputs for reporting.

**Technologies Used**

Python (NumPy, pathlib)

scikit-learn (fetch_openml, SGDClassifier, cross_val_score, cross_val_predict, metrics)

Matplotlib (visualization & saving figures)

Jupyter / Google Colab environment

**Supported Classes**

Binary detection: digit 5 vs. not 5
(Underlying dataset contains digits 0–9 with 70,000 samples and 784 features per image.)

**How It Works**

Load the MNIST dataset from OpenML (mnist_784) and split data and targets.

Visualize a sample image by reshaping the 784 feature vector into a 28×28 array and plotting with Matplotlib.

Create binary target vectors (y_train_5, y_test_5) where label is True for digit 5.

Train an SGDClassifier on the training set to detect digit 5.

Evaluate the classifier using cross-validation accuracy and cross_val_predict.

Compute and interpret the confusion matrix, precision, and recall.

Compare performance against a DummyClassifier baseline.

**Usage**

Run the notebook in Google Colab or Jupyter:

Install any missing dependencies (scikit-learn, matplotlib) if required.

Execute cells sequentially to load data, visualize images, train the model, and run evaluations.

Inspect printed outputs such as cross-validation scores, confusion matrix, precision, and recall to assess classifier performance.

Save plots using the included save_fig() helper (saves to images/classification/).

**Example Code Snippets**
**Load dataset:**

from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', as_frame=False)
X, y = mnist["data"], mnist["target"]


**Visualize a digit:**

some_digit = X[0].reshape(28, 28)
plt.imshow(some_digit, cmap="binary")
plt.axis("off")
plt.show()


**Train binary SGD classifier:**

from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)


**Cross-validated prediction & metrics:**

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=5)
confusion_matrix(y_train_5, y_train_pred)
precision_score(y_train_5, y_train_pred)
recall_score(y_train_5, y_train_pred)


**Folder Structure**

mnist_classifier/
├── mnist_notebook.ipynb
├── images/
│ └── classification/
└── README.md

**Notes**

fetch_openml may emit a FutureWarning about parser defaults—this is informational; consider setting parser='auto' if needed in newer scikit-learn versions.

The notebook shows a binary classifier example for digit 5. Extend it to multiclass classification (0–9) using SGDClassifier with proper label encoding or other algorithms (Random Forest, CNN).

Accuracy can be misleading on imbalanced binary tasks; review precision, recall, and the confusion matrix.

For higher performance on MNIST, consider using convolutional neural networks (CNNs) with TensorFlow/Keras or PyTorch.

The included save_fig() helper saves figures to images/classification/ for documentation.
