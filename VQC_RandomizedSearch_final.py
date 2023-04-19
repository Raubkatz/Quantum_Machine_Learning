import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelBinarizer
from sklearn.base import BaseEstimator, ClassifierMixin
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.algorithms import VQC
from qiskit.algorithms.optimizers import SPSA
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes, PauliFeatureMap, ZFeatureMap, EfficientSU2, TwoLocal
from qiskit.algorithms.optimizers import COBYLA, SPSA, ADAM, L_BFGS_B, TNC, NFT
from datetime import datetime
import sys

def build_feature_map(feature_map_class, feature_dimension):
    if feature_map_class is None:
        return ZZFeatureMap(feature_dimension)
    else:
        return feature_map_class(feature_dimension)

def build_ansatz(ansatz_class, num_qubits):
    if ansatz_class is None:
        return RealAmplitudes(num_qubits=num_qubits)
    else:
        return ansatz_class(num_qubits=num_qubits)

class VQCWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, feature_map=None, ansatz=None, optimizer=None, quantum_instance=None, feature_dimension=4):
        self.feature_map = feature_map
        self.ansatz = ansatz
        self.optimizer = optimizer
        self.quantum_instance = quantum_instance
        self.feature_dimension = feature_dimension

    def _build_vqc(self):
        feature_map = build_feature_map(self.feature_map, self.feature_dimension)
        ansatz = build_ansatz(self.ansatz, feature_map.num_qubits)
        optimizer = self.optimizer or SPSA(maxiter=100)
        quantum_instance = self.quantum_instance or QuantumInstance(Aer.get_backend('qasm_simulator'))
        self.vqc = VQC(feature_map=feature_map, ansatz=ansatz, optimizer=optimizer, quantum_instance=quantum_instance)

    def fit(self, X, y):
        if self.feature_dimension is None:
            self.feature_dimension = X.shape[1]
        self._build_vqc()
        self.vqc.fit(X, y)
        return self

    def predict(self, X):
        return self.vqc.predict(X)

    def score(self, X, y, sample_weight=None):
        return self.vqc.score(X, y)

# Define the number of random picks
n_random_picks = 1

# Choose data set
data_nr = 0 #0: iris, 1: breast cancer, 2:

# Load and preprocess the Iris dataset
if data_nr == 0:
    data_sk = load_iris()
    data_name = "iris"
elif data_nr == 1:
    data_sk = load_breast_cancer()
    data_name = "breast cancer"
elif data_nr == 2:
    data_sk = load_wine()
    data_name = "wine"

else:
    print('No valid data choice, exiting...')
    sys.exit()

X = data_sk.data
y = data_sk.target

# Split the dataset
scaler = MinMaxScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
#scaler = MinMaxScaler().fit(X_train)
#X_train = scaler.transform(X_train)
#X_test = scaler.transform(X_test)

# Binarize labels
label_binarizer = LabelBinarizer()
y_train_onehot = label_binarizer.fit_transform(y_train)
y_test_onehot = label_binarizer.transform(y_test)

# Define the parameter grid for GridSearchCV
feature_maps = [PauliFeatureMap, ZFeatureMap, ZZFeatureMap]
ansatzes = [EfficientSU2, TwoLocal, RealAmplitudes]

param_grid = {
    'feature_map': feature_maps,
    'ansatz': ansatzes,
    'optimizer': [SPSA(maxiter=100), COBYLA(maxiter=100), SPSA(maxiter=100), ADAM(maxiter=100), L_BFGS_B(maxiter=100), NFT(maxiter=100)],
    'quantum_instance': [
        QuantumInstance(Aer.get_backend('qasm_simulator'), shots=1024),
        QuantumInstance(Aer.get_backend('statevector_simulator'))
    ]
}

# Perform RandomizedSearchCV
start = datetime.now()
random_search = RandomizedSearchCV(VQCWrapper(), param_grid, n_iter=n_random_picks, cv=5, random_state=42, verbose=10)
random_search.fit(X_train, y_train_onehot)

# Test the best model
print("Best parameters:", random_search.best_params_)
test_score = random_search.score(X_test, y_test_onehot)
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
gen_out_txt = "VQC_RandomizedSearchCV for the " + str(data_name) + " dataset with n_iter=" + str(n_random_picks) + "\n"
gen_out_txt = gen_out_txt + "CV-score:" + "\n" + str(random_search.best_score_) + "\n" + "test-score:" + "\n" + str(test_score) + "\n" + "runtime:" + "\n" + str(end) + "\n"
gen_out_txt = gen_out_txt + "best parameters:" + "\n" + str(random_search.best_params_) + "\n"
gen_out_txt = gen_out_txt + "RandomizedSearchCV.cv_results:" + "\n" + str(random_search.cv_results_)

#write CV stats file
InfoPath = "./VQC_RandomizedSearchCV_" + data_name + "_" + str(n_random_picks) + ".txt"
InfoFile = open(InfoPath, "w+")
InfoFile.write(gen_out_txt)
InfoFile.close()
