
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelBinarizer
from sklearn.datasets import fetch_openml
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.base import BaseEstimator, ClassifierMixin
from qiskit import Aer
from sklearn.decomposition import PCA
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.algorithms import VQC
from qiskit.algorithms.optimizers import SPSA
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes, PauliFeatureMap, ZFeatureMap, EfficientSU2, TwoLocal
from qiskit.algorithms.optimizers import COBYLA, SPSA, ADAM, L_BFGS_B, TNC, NFT
from datetime import datetime
from copy import deepcopy as dc
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit.circuit.library import ZFeatureMap, PauliFeatureMap, ZZFeatureMap, RealAmplitudes
from sklearn.base import BaseEstimator, ClassifierMixin
import sys
import qiskit
import qiskit_machine_learning
print('qiskit and quiskit machine learning versions:')
print(qiskit.__version__)
print(qiskit_machine_learning.__version__)

#gpu
#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
# Define the number of random picks
n_random_picks = 2
cv = 5
nr_pca = 20

# Choose data set
data_nr = 1 # 0: iris, 1: breast cancer, 2: wine data set

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
    data_sk = fetch_openml(name='ilpd', version=1, as_frame=True)
    data_name = "ilpd"
elif data_nr == 4:
    data_sk = fetch_openml(name='phoneme', version=1, as_frame=True)
    data_name = "phoneme"
elif data_nr == 5:
    data_sk = fetch_openml(name='Insurance', version=1, as_frame=True)
    data_name = "Insurance"
else:
    print('No valid data choice, exiting...')
    sys.exit()

X = data_sk.data
y = data_sk.target

if data_nr >= 3:  # we need to do additional preprocessing for the adult data set

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

print("Starting Randomized Search for the " + str(data_name) + " data set,\n" + str(n_random_picks) + " randomly parameterized models will be tested in a " + str(cv) +"-fold cross validation.\n\n")
if X.shape[1] > 20:  # we need to use PCA to reduce the number of features
    print("Too man features, PCA will be applied.")
    # Apply PCA
    pca = PCA(n_components=nr_pca)  # Keep 95% of the variance
    X = pca.fit_transform(X)
    data_name = data_name + "_PCA"

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class QKEWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, feature_map=None, quantum_instance=None, feature_dimension=None, C=1.0):
        self.feature_map = feature_map
        self.quantum_instance = quantum_instance
        self.feature_dimension = feature_dimension
        self.X_train_ = None
        self.C = C

    def _build_feature_map(self, feature_map_class, feature_dimension):
        if feature_map_class is None:
            return ZFeatureMap(feature_dimension)
        else:
            return feature_map_class(feature_dimension)

    def _build_quantum_kernel(self):
        self.feature_map = self._build_feature_map(self.feature_map, self.feature_dimension)
        if self.quantum_instance is None:
            self.quantum_instance = QuantumInstance(Aer.get_backend('qasm_simulator'), shots=1024)

        self.q_kernel = QuantumKernel(feature_map=self.feature_map, quantum_instance=self.quantum_instance)

    def fit(self, X, y):
        self.X_train_ = X
        if self.feature_dimension is None:
            self.feature_dimension = X.shape[1]
        self._build_quantum_kernel()
        self.kernel_matrix_train = self.q_kernel.evaluate(x_vec=X)
        self.svc = SVC(kernel='precomputed', C=self.C).fit(self.kernel_matrix_train, y)
        self.support_vectors_ = X[self.svc.support_]
        return self

    def predict(self, X):
        kernel_matrix_test = self.q_kernel.evaluate(x_vec=X, y_vec=self.X_train_)
        return self.svc.predict(kernel_matrix_test)

    def score(self, X, y, sample_weight=None):
        kernel_matrix_test = self.q_kernel.evaluate(x_vec=X, y_vec=self.X_train_)
        return self.svc.score(kernel_matrix_test, y)

# Define the parameter grid for GridSearchCV
feature_maps = [ZFeatureMap, PauliFeatureMap, ZZFeatureMap]

quantum_instances = [
    QuantumInstance(Aer.get_backend('aer_simulator'), shots=1024),
    QuantumInstance(Aer.get_backend('qasm_simulator'), shots=1024)]
#C_values = np.logspace(-4, 4, 9)
C_values = np.logspace(-3, 3, 9)

param_grid = {
    'feature_map': feature_maps,
    'quantum_instance': quantum_instances,
    'C': C_values,
}


# Perform RandomizedSearchCV
start = datetime.now()
random_search = RandomizedSearchCV(QKEWrapper(), param_grid, n_iter=n_random_picks, cv=cv, random_state=42, verbose=10)
random_search.fit(X_train, y_train)

# Test the best model
print("Best parameters:", random_search.best_params_)
test_score = random_search.score(X_test, y_test)
print("Test accuracy:", test_score)

# calculate runtime
now = datetime.now()
end = now - start

# print runtime
print('#################')
print('Finished: ', now)
print('Runtime: ', end)
print('#################')

# generate CV stats file
gen_out_txt = "QKE_RandomizedSearchCV for the " + str(data_name) + " dataset with n_iter=" + str(n_random_picks) + "\n"
gen_out_txt = gen_out_txt + "CV-score:" + "\n" + str(random_search.best_score_) + "\n" + "test-score:" + "\n" + str \
    (test_score) + "\n" + "runtime:" + "\n" + str(end) + "\n"
gen_out_txt = gen_out_txt + "best parameters:" + "\n" + str(random_search.best_params_) + "\n"
gen_out_txt = gen_out_txt + "RandomizedSearchCV.cv_results:" + "\n" + str(random_search.cv_results_)

# write CV stats file
InfoPath = "./QKE_RandomizedSearchCV_" + data_name + "_" + str(n_random_picks) + ".txt"
InfoFile = open(InfoPath, "w+")
InfoFile.write(gen_out_txt)
InfoFile.close()