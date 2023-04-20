import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import time
import os
import datetime
from sklearn.decomposition import PCA
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit import BasicAer
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit import Aer
from qiskit.circuit.library import ZFeatureMap, PauliFeatureMap, ZZFeatureMap, RealAmplitudes

if not os.path.exists("results"):
    os.makedirs("results")



##################### Classic ML Comparison #############################

sample_sizes = np.arange(500, 2001, 500)
classifiers = {
    'SVM': SVC(),
    'XGBoost': XGBClassifier(eval_metric='logloss', use_label_encoder=False),
    #'MLP': MLPClassifier(random_state=42)
}

# Set PCA to "Yes" or "No"
use_PCA = "Yes"


results = {key: {'times': [], 'accuracies': []} for key in classifiers.keys()}

for sample_size in sample_sizes:
    print(f"Classifying with {sample_size} samples")

    # Generate artificial dataset
    X, y = make_classification(n_samples=sample_size, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=42)

    if use_PCA == "Yes":
        # Perform PCA on the generated data
        pca = PCA(n_components=2)
        X = pca.fit_transform(X)


    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for clf_name, clf in classifiers.items():
        # Train and evaluate the classifier
        start_time = time.time()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        end_time = time.time()

        # Calculate the time taken and classification accuracy
        time_taken = end_time - start_time
        results[clf_name]['times'].append(time_taken)

        accuracy = accuracy_score(y_test, y_pred)
        results[clf_name]['accuracies'].append(accuracy)

        print(f"{clf_name}: Time taken for {sample_size} samples: {time_taken:.2f} seconds, Accuracy: {accuracy:.4f}")

    print('\n')

# Save the results as a text file
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
pca_prefix = "PCA_" if use_PCA == "Yes" else ""
for clf_name, data in results.items():
    with open(f"results/artificial_runtime_exp_{pca_prefix}{clf_name}_{timestamp}.txt", "w") as f:
        for sample_size, time_taken, accuracy in zip(sample_sizes, data['times'], data['accuracies']):
            f.write(f"Sample size: {sample_size}, Time taken: {time_taken:.2f} seconds, Accuracy: {accuracy:.4f}\n")


# Plot the graphs comparing runtimes and accuracies of all iterations
fig, axs = plt.subplots(2, len(classifiers), figsize=(len(classifiers) * 6, 10), sharex=True)

for i, (clf_name, data) in enumerate(results.items()):
    axs[0, i].plot(sample_sizes, data['times'], color='tab:blue')
    axs[0, i].set_title(clf_name)
    axs[0, i].set_ylabel('Runtime (seconds)')
    axs[0, i].grid()

    axs[1, i].plot(sample_sizes, data['accuracies'], color='tab:red')
    axs[1, i].set_xlabel('Number of Samples')
    axs[1, i].set_ylabel('Accuracy')
    axs[1, i].grid()

plt.suptitle('Runtime and Accuracy vs Number of Samples for Classical ML Classifiers')

# Save the plots as png files in the results folder
for i, clf_name in enumerate(classifiers.keys()):
    plt.savefig(f"results/artificial_runtime_exp_{clf_name}_{timestamp}.png")

plt.show()

# Save the plots as png files in the results folder
for i, clf_name in enumerate(classifiers.keys()):
    plt.savefig(f"results/artificial_runtime_exp_{pca_prefix}{clf_name}_{timestamp}.png")

plt.show()


