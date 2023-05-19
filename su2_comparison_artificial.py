import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report

# Define the Pauli matrices
sigma_1 = np.array([[0, 1], [1, 0]])
sigma_2 = np.array([[0, -1j], [1j, 0]])
sigma_3 = np.array([[1, 0], [0, -1]])

# Define a function to generate a SU(2) matrix
def generate_su2(theta, phi, lam):
    return np.cos(theta/2) * np.identity(2) - 1j * np.sin(theta/2) * (np.sin(phi) * sigma_1 + np.cos(phi) * sigma_2 + np.sin(lam) * sigma_3)

# Generate the dataset
n_samples = 1000
data = []
for _ in range(n_samples):
    theta = np.random.uniform(0, np.pi)
    phi = np.random.uniform(0, 2 * np.pi)
    lam = np.random.uniform(0, 2 * np.pi)
    matrix = generate_su2(theta, phi, lam)
    features = list(matrix.flatten().real) + list(matrix.flatten().imag)
    top_right_element = matrix[0, 1].real
    label = 0 if top_right_element > 0 else 1
    print(label)
    data.append(features + [label])

# Create a DataFrame
df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(1, 9)] + ['label'])

# Split the dataset into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Separate the features from the labels
X_train = train_df.drop('label', axis=1)
y_train = train_df['label']
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

# Train a SVM classifier
clf = svm.SVC()
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))

