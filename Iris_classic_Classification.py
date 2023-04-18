import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.linear_model import Lasso, Ridge, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import lightgbm as lgb
import time

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features for better performance
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define classifiers and their hyperparameters for randomized search
classifiers = {
    'SVM': (SVC(random_state=42), {
        'C': np.logspace(-3, 3, 7),
        'kernel': ['linear'],
        'degree': [2],
        'coef0': np.linspace(-1, 3, 7),
    }),
    'Lasso': (LogisticRegression(penalty='l1', solver='saga', random_state=42), {
        'C': np.logspace(-3, 3, 7),
    }),
    'Ridge': (LogisticRegression(penalty='l2', random_state=42), {
        'C': np.logspace(-3, 3, 7),
    }),
    'XGBoost': (xgb.XGBClassifier(eval_metric='mlogloss', random_state=42), {
        'n_estimators': [50],
        'max_depth': [3, 4],
        'learning_rate': np.logspace(-3, 0, 4),
        'subsample': [0.5],
        'colsample_bytree': [0.5],
    }),
    'LightGBM': (lgb.LGBMClassifier(random_state=42), {
        'n_estimators': [50],
        'max_depth': [3, 4],
        'learning_rate': np.logspace(-3, 0, 4),
        'subsample': [0.5],
        'colsample_bytree': [0.5],
    }),
    'MLP': (MLPClassifier(random_state=42), {
    'hidden_layer_sizes': [(50,)],
    'activation': ['identity', 'logistic'],
    'solver': ['lbfgs'],
    'alpha': np.logspace(-5, 0, 4),
    'learning_rate': ['constant'],
}),
            }

# Run randomized search with 5-fold cross-validation for each classifier
best_params = {}
best_scores = {}
trained_models = {}

for name, (classifier, params) in classifiers.items():
    print(f"Performing randomized search for {name}...")
    start_time = time.time()

    random_search = RandomizedSearchCV(
        classifier, param_distributions=params, n_iter=50, cv=5, n_jobs=-1, random_state=42
    )
    random_search.fit(X_train_scaled, y_train)

    best_params[name] = random_search.best_params_
    best_scores[name] = random_search.best_score_

    print(f"{name} best parameters: {best_params[name]}")
    print(f"{name} best mean cross-validated score: {best_scores[name]:.4f}")

    # Train the best model found and evaluate it on the test set
    best_model = random_search.best_estimator_
    best_model.fit(X_train_scaled, y_train)
    y_pred = best_model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_pred)

    print(f"{name} test set accuracy: {test_accuracy:.4f}")

    # Print classification report
    report = classification_report(y_test, y_pred)
    print(f"{name} classification report:\n{report}")


    print(f"Time taken: {time.time() - start_time:.2f} seconds\n")

    # Store the trained best model
    trained_models[name] = best_model

