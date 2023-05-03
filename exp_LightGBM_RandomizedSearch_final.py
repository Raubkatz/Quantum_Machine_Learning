import numpy as np
import lightgbm as lgb
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.datasets import fetch_openml
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from datetime import datetime
from copy import deepcopy as dc
import sys
import os

# Define the number of random picks
n_random_picks = 20
cv = 5
nr_pca = 20

# Choose data set
data_nr = 0

# Load the data set
if data_nr == 0: # in the paper
    data_sk = load_iris()
    data_name = "iris"
elif data_nr == 1: # not in the paper
    data_sk = load_breast_cancer()
    data_name = "breast cancer"
elif data_nr == 2: # in the paper
    data_sk = load_wine()
    data_name = "wine"
elif data_nr == 3: # in the paper
    data_sk = fetch_openml(name='ilpd', version=1, as_frame=True)
    data_name = "ilpd"
elif data_nr == 4: # not in the paper
    data_sk = fetch_openml(name='phoneme', version=1, as_frame=True)
    data_name = "phoneme"
elif data_nr == 5: # not in the paper
    data_sk = fetch_openml(name='Insurance', version=1, as_frame=True)
    data_name = "Insurance"
elif data_nr == 6: # in the paper
    data_sk = fetch_openml(name='breast-cancer-coimbra', version=1, as_frame=True)
    data_name = "breast-cancer-coimbra"
elif data_nr == 7: # not in the paper
    data_sk = fetch_openml(name='Is-this-a-good-customer', version=1, as_frame=True)
    data_name = "Is-this-a-good-customer"
elif data_nr == 8: # in the paper
    data_sk = fetch_openml(name='tae', version=1, as_frame=True)
    data_name = "tae"
elif data_nr == 9: # in the paper
    data_sk = fetch_openml(name='breast-tissue', version=1, as_frame=True)
    data_name = "breast-tissue"
else:
    print('No valid data choice, exiting...')
    sys.exit()

X = data_sk.data
y = data_sk.target

if data_nr >= 3:  # we need to do additional preprocessing for the datsets featuring categorical data

    # Identify categorical and continuous features
    cat_features = X.select_dtypes(include=['object']).columns
    cont_features = X.select_dtypes(include=[np.number]).columns

    # Create a ColumnTransformer instance with OneHotEncoder for categorical features and MinMaxScaler for continuous features
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), cat_features),
            ('cont', MinMaxScaler(), cont_features)
        ])

    # Preprocess the features
    X = dc(preprocessor.fit_transform(X))

else:
    scaler = MinMaxScaler().fit_transform(X)

pid = os.getpid() # get the process ID
pgid = os.getpgid(pid) # get the process group ID (PGID)
print("Starting Grid Search for the " + str(data_name) + " data set,\n" + str(n_random_picks) + " randomly parameterized models will be tested in a " + str(cv) +"-fold cross validation.\n\n")
print("Job ID:", pgid)

if X.shape[1] > 20:  # we need to use PCA to reduce the number of features
    print("Too man features, PCA will be applied.")
    # Apply PCA
    pca = PCA(n_components=nr_pca)  # Keep 95% of the variance
    X = pca.fit_transform(X)
    data_name = data_name + "_PCA"

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for RandomizedSearchCV
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [50, 100, 150, 200],
    'subsample': [0.5, 0.8, 1],
    'colsample_bytree': [0.5, 0.8, 1]
}

# Perform RandomizedSearchCV
start = datetime.now()
random_search = RandomizedSearchCV(lgb.LGBMClassifier(), param_grid, n_iter=n_random_picks, cv=cv, random_state=42, verbose=10)
random_search.fit(X_train, y_train)

# Test the best model
print("Best parameters:", random_search.best_params_)
test_score = random_search.score(X_test, y_test)
print("Test accuracy:", test_score)

# Calculate runtime
now = datetime.now()
end = now - start

# Print runtime
print('#################')
print('Finished: ', now)
print('Runtime: ', end)
print('#################')

# Generate CV stats file
gen_out_txt = "LightGBM_RandomizedSearchCV for the " + str(data_name) + " dataset with n_iter=" + str(n_random_picks) + "\n"
gen_out_txt = gen_out_txt + "CV-score:" + "\n" + str(random_search.best_score_) + "\n" + "test-score:" + "\n" + str(test_score) + "\n" + "runtime:" + "\n" + str(end) + "\n"
gen_out_txt = gen_out_txt + "best parameters:" + "\n" + str(random_search.best_params_) + "\n"
gen_out_txt = gen_out_txt + "RandomizedSearchCV.cv_results:" + "\n" + str(random_search.cv_results_)

# Write CV stats file
InfoPath = "./LightGBM_RandomizedSearchCV_" + data_name + "_" + str(n_random_picks) + ".txt"
InfoFile = open(InfoPath, "w+")
InfoFile.write(gen_out_txt)
InfoFile.close()
