import numpy as np
import xgboost as xgb
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelBinarizer
from sklearn.datasets import fetch_openml
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from datetime import datetime
from sklearn.decomposition import PCA
from copy import deepcopy as dc
import sys

# Define the number of random picks
n_random_picks = 1
switch_PCA = True

# Choose data set
data_nr = 0 #0: iris, 1: breast cancer, 2: wine data set, 3: adult data set aka census income

# Load and preprocess the Iris dataset, all features of these datasets are conitnuous no additional preprocessing required
if data_nr == 0:
    data_sk = load_iris()
    data_name = "iris"
elif data_nr == 1:
    data_sk = load_breast_cancer()
    data_name = "breast cancer"
elif data_nr == 2:
    data_sk = load_wine()
    data_name = "wine"
elif data_nr == 3:
    data_sk = fetch_openml(name='adult', version=1, as_frame=True)
    data_name = "adult"
else:
    print('No valid data choice, exiting...')
    sys.exit()

X = data_sk.data
y = data_sk.target

if data_nr == 3: #we need to do additional preprocessing for the adult data set

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

if switch_PCA:
    # Apply PCA
    pca = PCA(n_components=0.95)  # Keep 95% of the variance
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
random_search = RandomizedSearchCV(xgb.XGBClassifier(), param_grid, n_iter=n_random_picks, cv=5, random_state=42, verbose=10)
random_search.fit(X_train, y_train)

# Test the best model
print("Best parameters:", random_search.best_params_)
test_score = random_search.score(X_test, y_test)
print("Test accuracy:", test_score)

#calculate runtime
now = datetime.now()
end = now - start

#print runtime
print('#################')
print('Finished: ', now)
print('Runtime: ', end)
print('#################')

#generate CV stats file

#generate CV stats file
gen_out_txt = "XGBoost_RandomizedSearchCV for the " + str(data_name) + " dataset with n_iter=" + str(n_random_picks) + "\n"
gen_out_txt = gen_out_txt + "CV-score:" + "\n" + str(random_search.best_score_) + "\n" + "test-score:" + "\n" + str(test_score) + "\n" + "runtime:" + "\n" + str(end) + "\n"
gen_out_txt = gen_out_txt + "best parameters:" + "\n" + str(random_search.best_params_) + "\n"
gen_out_txt = gen_out_txt + "RandomizedSearchCV.cv_results:" + "\n" + str(random_search.cv_results_)

#write CV stats file
InfoPath = "./XGBoost_RandomizedSearchCV_" + data_name + "_" + str(n_random_picks) + ".txt"
InfoFile = open(InfoPath, "w+")
InfoFile.write(gen_out_txt)
InfoFile.close()