"""
Created by Sebastian Raubitzek and Kevin Mallinger (SBA Research and TU Wien) as part of the publication "On the Applicability of Quantum Machine Learning".

DOI: 10.1234/fantasy-doi.2023.04.20
Date: 20.04.2023

This program provides a wrapper for Qiskit Machine Learning's Variational Quantum Classifier (VQC) and demonstrates how to use scikit-learn's RandomizedSearchCV to find a suitable set of hyperparameters for the VQC model.
"""
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelBinarizer, LabelEncoder
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
import sys
import qiskit
import qiskit_machine_learning
import os
print('qiskit and quiskit machine learning versions:')
print(qiskit.__version__)
print(qiskit_machine_learning.__version__)

def build_feature_map(feature_map_class, feature_dimension):
    """
    Builds the feature map based on the provided parameter

    :param feature_map_class: The feature map class to be used.
    :type feature_map_class: FeatureMap or None
    :param feature_dimension: The dimensionality of the feature space.
    :type feature_dimension: int
    :return: An instance of the feature map.
    :rtype: FeatureMap
    """
    if feature_map_class is None:
        return ZZFeatureMap(feature_dimension)
    else:
        return feature_map_class(feature_dimension)

def build_ansatz(ansatz_class, num_qubits):
    """
    Builds the ansatz (variational form) based on the provided parameters.

    :param ansatz_class: The ansatz class to be used.
    :type ansatz_class: VariationalForm or None
    :param num_qubits: The number of qubits for the ansatz.
    :type num_qubits: int
    :return: An instance of the ansatz.
    :rtype: VariationalForm
    """
    if ansatz_class is None:
        return RealAmplitudes(num_qubits=num_qubits)
    else:
        return ansatz_class(num_qubits=num_qubits)

class VQCWrapper(BaseEstimator, ClassifierMixin):
    """
    VQCWrapper is a wrapper class for Variational Quantum Classifier (VQC). It uses a variational
    quantum circuit to perform binary classification tasks. Inherits from BaseEstimator and ClassifierMixin.
    """

    def __init__(self, feature_map=None, ansatz=None, optimizer=None, quantum_instance=None, feature_dimension=None):
        """
        Initializes the VQCWrapper class with the provided parameters.

        :param feature_map: The feature map to be used for VQC. If not provided, ZZFeatureMap will be used.
        :type feature_map: FeatureMap or None
        :param ansatz: The ansatz (variational form) to be used for VQC. If not provided, RealAmplitudes will be used.
        :type ansatz: VariationalForm or None
        :param optimizer: The optimizer to be used for VQC. If not provided, SPSA with maxiter=100 will be used.
        :type optimizer: Optimizer or None
        :param quantum_instance: The quantum instance to be used for VQC. If not provided, a default instance with Aer's qasm_simulator backend will be created.
        :type quantum_instance: QuantumInstance or None
        :param feature_dimension: The dimensionality of the feature space. If not provided, it will be inferred from the input data during fitting.
        :type feature_dimension: int or None
        """
        self.feature_map = feature_map
        self.ansatz = ansatz
        self.optimizer = optimizer
        self.quantum_instance = quantum_instance
        self.feature_dimension = feature_dimension

    def _build_vqc(self):
        """
        Builds the VQC using the feature map, ansatz, optimizer, and quantum instance.
        """
        feature_map = build_feature_map(self.feature_map, self.feature_dimension)
        ansatz = build_ansatz(self.ansatz, feature_map.num_qubits)
        optimizer = self.optimizer or SPSA(maxiter=100)
        quantum_instance = self.quantum_instance or QuantumInstance(Aer.get_backend('qasm_simulator'))
        self.vqc = VQC(feature_map=feature_map, ansatz=ansatz, optimizer=optimizer, quantum_instance=quantum_instance)

    def fit(self, X, y):
        """
        Fits the VQCWrapper on the provided training data.

        :param X: Training input samples.
        :type X: array-like
        :param y: Training target values.
        :type y: array-like
        :return: The fitted estimator.
        :rtype: self
        """
        if self.feature_dimension is None:
            self.feature_dimension = X.shape[1]
        self._build_vqc()
        self.vqc.fit(X, y)
        return self

    def predict(self, X):
        """
        Performs predictions on the provided input samples.

        :param X: Input samples.
        :type X: array-like
        :return: Predicted class labels.
        :rtype: array-like
        """
        return self.vqc.predict(X)

    def score(self, X, y, sample_weight=None):
        """
        Computes the mean accuracy on the provided test data and labels.

        :param X: Test input samples.
        :type X: array-like
        :param y: Test target values.
        :type y: array-like
        :param sample_weight: Optional sample weights. If not provided, the default is None.
        :type sample_weight: array-like or None
        :return: Mean accuracy of the estimator.
        :rtype: float
        """
        return self.vqc.score(X, y)

# Define the number of random picks
n_random_picks = 20
cv = 5
switch_PCA = False
nr_pca = 20
max_iter = 100# maximum of iterations for the optimizer

# Choose data set
data_nr = 0 # 0: iris, 1: breast cancer, 2: wine data set, 3:

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
elif data_nr == 8: # in the paper
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
print("Starting VQC Grid Search for the " + str(data_name) + " data set,\n" + str(n_random_picks) + " randomly parameterized models will be tested in a " + str(cv) +"-fold cross validation.\n\n")
print("Job ID:", pgid)

if X.shape[1] > 20:  # we need to use PCA to reduce the number of features
    print("Too man features, PCA will be applied.")
    # Apply PCA
    pca = PCA(n_components=nr_pca)  # Keep 95% of the variance
    X = pca.fit_transform(X)
    data_name = data_name + "_PCA"

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Binarize labels if there are more than two classes in y_train
if len(np.unique(y_train)) == 2:
    try:
        # Transform target variable into binary matrix
        encoder = OneHotEncoder(sparse=False)
        y_encoded = encoder.fit_transform(y.reshape(-1, 1))
        y_train_onehot = encoder.fit_transform(y_train.reshape(-1, 1))
        y_test_onehot = encoder.fit_transform(y_test.reshape(-1, 1))
    except AttributeError:
        # If y is a Pandas Series object, convert to numpy array and then reshape
        y_numpy = y.to_numpy().reshape(-1, 1)
        encoder = OneHotEncoder(sparse=False)
        y_encoded = encoder.fit_transform(y_numpy)
        y_train_onehot = encoder.fit_transform(y_train.to_numpy().reshape(-1, 1))
        y_test_onehot = encoder.fit_transform(y_test.to_numpy().reshape(-1, 1))
else:
    label_binarizer = LabelBinarizer()
    y_train_onehot = label_binarizer.fit_transform(y_train)
    y_test_onehot = label_binarizer.transform(y_test)

# Define the parameter grid for GridSearchCV
feature_maps = [PauliFeatureMap, ZFeatureMap, ZZFeatureMap]
ansatzes = [EfficientSU2, TwoLocal, RealAmplitudes]

param_grid = {
    'feature_map': [PauliFeatureMap, ZFeatureMap, ZZFeatureMap],
    'ansatz': [EfficientSU2, TwoLocal, RealAmplitudes],
    'optimizer': [
        COBYLA(maxiter=max_iter),
        SPSA(maxiter=max_iter),
        NFT(maxiter=max_iter),
    ],
    'quantum_instance': [
        QuantumInstance(Aer.get_backend('aer_simulator'), shots=1024),
        QuantumInstance(Aer.get_backend('qasm_simulator'), shots=1024),
        QuantumInstance(Aer.get_backend('statevector_simulator'), shots=1024) #maybe delete this
    ],
}

# Perform RandomizedSearchCV
start = datetime.now()
random_search = RandomizedSearchCV(VQCWrapper(), param_grid, n_iter=n_random_picks, cv=cv, random_state=42, verbose=10)
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