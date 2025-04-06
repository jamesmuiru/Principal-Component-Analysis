# Principal-Component-Analysis
This code acts as a guide on how to do principal component analysis
Principal Component Analysis (PCA) on Breast Cancer Dataset
===========================================================

This project demonstrates how to perform Principal Component Analysis (PCA)
on the breast cancer dataset from scikit-learn. It includes visualization 
and identification of the most influential features.

Steps Included
--------------

1. Load the dataset using sklearn.
2. Standardize the features.
3. Apply PCA to reduce dimensions to 2.
4. Plot the PCA-transformed data.
5. Display the top contributing features to PC1 and PC2.

Requirements
------------

- Python 3.x
- pandas
- matplotlib
- seaborn
- scikit-learn

Install packages using:

    pip install pandas matplotlib seaborn scikit-learn

How to Run
----------

1. Run the Python script (e.g., `pca_example.py`) containing the following logic:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Step 2: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Step 4: Plot PCA results
plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y, palette="coolwarm", alpha=0.7)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA - Breast Cancer Dataset")
plt.show()
