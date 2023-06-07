from xgboost import XGBClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier  # Import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
import os
import random
from sklearn.decomposition import PCA
import time
from func_su2_su3_rand_data import *

# Create results folder if it doesn't exist
if not os.path.exists('results_sun'):
    os.makedirs('results_sun')

sample_sizes = [50, 100, 250, 500, 1000, 1500, 2000]
random.seed(42)
n_random_picks = 20

use_randomized_search = False  # Set this to False for out-of-the-box implementation
search_prefix = "RandomSearch_" if use_randomized_search else "OutOfTheBox_"
use_PCA = "No"
pca_prefix = "PCA_" if use_PCA == "Yes" else ""


# Define the parameter grid for RandomizedSearchCV
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [50, 100, 150, 200],
    'subsample': [0.5, 0.8, 1],
    'colsample_bytree': [0.5, 0.8, 1]
}

times = []
accuracies = []


for sample_size in sample_sizes:
    print(f"Classifying with {sample_size} samples")

    # Generate artificial dataset
    print(f"Generating su2 data set")
    data = dc(generate_su2_data(sample_size, binary=True, c_decision=2))
    X, y = dc(prepare_classification_data(data))

    if use_PCA == "Yes":
        # Perform PCA on the generated data
        pca = PCA(n_components=2)
        X = pca.fit_transform(X)

    # Normalize and rescale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create the XGBoost classifier
    xgb = XGBClassifier(random_state=42)

    if use_randomized_search:
        # RandomizedSearchCV object
        random_search = RandomizedSearchCV(xgb, param_distributions=param_grid, n_iter=n_random_picks, cv=5, n_jobs=-1)
        search_model = random_search
    else:
        search_model = xgb

    # Train the classifier and record the start time
    start_time = time.time()
    search_model.fit(X_train, y_train)
    end_time = time.time()

    # Predict the labels and calculate the accuracy
    y_pred = search_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Calculate the time taken
    time_taken = end_time - start_time

    print(f"Time taken for {sample_size} samples: {time_taken:.2f} seconds")
    print(f"Accuracy for {sample_size} samples: {accuracy:.2f}\n")
    if use_randomized_search:
        print(f"Best params: {random_search.best_params_}\n")

    times.append(time_taken)
    accuracies.append(accuracy)

    # Save results in a single text file
    results_array = np.column_stack((sample_size, time_taken, accuracy))
    #np.savetxt(f"results/{pca_prefix}{search_prefix}XGBoost_results_{sample_size}.txt", results_array, fmt='%.6f',
    #           header='Sample_Size Time(s) Accuracy', delimiter='\t', comments='')
results_array = np.column_stack((sample_sizes, times, accuracies))
np.savetxt(f"results_sun/{pca_prefix}{search_prefix}XGBoost_results.txt", results_array, fmt='%.6f',
           header='Sample_Size Time(s) Accuracy', delimiter='\t', comments='')
