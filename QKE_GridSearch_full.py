"""
Created by Sebastian Raubitzek and Kevin Mallinger (SBA Research and TU Wien) as part of the publication "On the Applicability of Quantum Machine Learning".

DOI: 10.1234/fantasy-doi.2023.04.20
Date: 20.04.2023

This program provides a wrapper for Qiskit Machine Learning's Quantum Kernel Estimator (QKE) and demonstrates how to use scikit-learn's GridSearchCV to find a suitable set of hyperparameters for a QKE model.
"""

import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
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
import os
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
class QKEWrapper(BaseEstimator, ClassifierMixin):
    """
    QKEWrapper is a wrapper class for Quantum Kernel Estimator (QKE). It uses quantum kernel methods to
    perform classification tasks. Inherits from BaseEstimator and ClassifierMixin.
    """

    def __init__(self, feature_map=None, quantum_instance=None, feature_dimension=None, C=1.0):
        """
        Initializes the QKEWrapper class with the provided parameters.

        :param feature_map: The feature map to be used for quantum kernel. If not provided, ZFeatureMap will be used.
        :type feature_map: FeatureMap or None
        :param quantum_instance: The quantum instance to be used for the quantum kernel. If not provided, a default
                                 instance with Aer's qasm_simulator backend will be created.
        :type quantum_instance: QuantumInstance or None
        :param feature_dimension: The dimensionality of the feature space. If not provided, it will be inferred from
                                  the input data during fitting.
        :type feature_dimension: int or None
        :param C: Regularization parameter for the SVM. Default is 1.0.
        :type C: float
        """
        self.feature_map = feature_map
        self.quantum_instance = quantum_instance
        self.feature_dimension = feature_dimension
        self.X_train_ = None
        self.C = C


    def _build_feature_map(self, feature_map_class, feature_dimension):
        """
        Builds the feature map based on the provided parameters.

        :param feature_map_class: The feature map class to be used.
        :type feature_map_class: FeatureMap or None
        :param feature_dimension: The dimensionality of the feature space.
        :type feature_dimension: int
        :return: An instance of the feature map.
        :rtype: FeatureMap
        """
        if feature_map_class is None:
            return ZFeatureMap(feature_dimension)
        else:
            return feature_map_class(feature_dimension)

    def _build_quantum_kernel(self):
        """
        Builds the quantum kernel using the feature map and quantum instance.
        """
        self.feature_map = self._build_feature_map(self.feature_map, self.feature_dimension)
        if self.quantum_instance is None:
            self.quantum_instance = QuantumInstance(Aer.get_backend('qasm_simulator'), shots=1024)

        self.q_kernel = QuantumKernel(feature_map=self.feature_map, quantum_instance=self.quantum_instance)

    def fit(self, X, y):
        """
        Fits the QKEWrapper on the provided training data.

        :param X: Training input samples.
        :type X: array-like
        :param y: Training target values.
        :type y: array-like
        :return: The fitted estimator.
        :rtype: self
        """
        self.X_train_ = X
        if self.feature_dimension is None:
            self.feature_dimension = X.shape[1]
        self._build_quantum_kernel()
        self.kernel_matrix_train = self.q_kernel.evaluate(x_vec=X)
        self.svc = SVC(kernel='precomputed', C=self.C).fit(self.kernel_matrix_train, y)
        self.support_vectors_ = X[self.svc.support_]
        return self

    def predict(self, X):
        """
        Performs predictions on the provided input samples.

        :param X: Input samples.
        :type X: array-like
        :return: Predicted class labels.
        :rtype: array-like
        """
        kernel_matrix_test = self.q_kernel.evaluate(x_vec=X, y_vec=self.X_train_)
        return self.svc.predict(kernel_matrix_test)

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
        kernel_matrix_test = self.q_kernel.evaluate(x_vec=X, y_vec=self.X_train_)
        return self.svc.score(kernel_matrix_test, y)

# Define the number the grid search
cv = 5 #nr cross validation
nr_pca = 20 #in case of a PCA, what are the number of features the data should be reduced to.

# Choose data set
data_nr = 0 # 0: iris, 1: breast cancer, 2: wine data set

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

pid = os.getpid() # get the process ID
pgid = os.getpgid(pid) # get the process group ID (PGID)
start = datetime.now()

print("Starting QKE Grid Search for the " + str(data_name) + " data set,\n" + "Several models parameterized using a parameter grid will be tested in a " + str(cv) +"-fold cross validation.\n\n")
print("Job ID:", pgid)
print('Start time: ', start)

if X.shape[1] > 20:  # we need to use PCA to reduce the number of features
    print("Too man features, PCA will be applied.")
    # Apply PCA
    pca = PCA(n_components=nr_pca)  # Keep 95% of the variance
    X = pca.fit_transform(X)
    data_name = data_name + "_PCA"

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for GridSearchCV
feature_maps = [ZFeatureMap, PauliFeatureMap, ZZFeatureMap]

quantum_instances = [
    QuantumInstance(Aer.get_backend('aer_simulator'), shots=1024),
    QuantumInstance(Aer.get_backend('qasm_simulator'), shots=1024),
    QuantumInstance(Aer.get_backend('statevector_simulator'), shots=1024) #maybe delete this
    ]
#C_values = np.logspace(-4, 4, 9)
C_values = np.logspace(-3, 3, 9)

param_grid = {
    'feature_map': feature_maps,
    'quantum_instance': quantum_instances,
    'C': C_values,
}

# Perform GridSearchCV
grid_search = GridSearchCV(QKEWrapper(), param_grid, cv=cv, verbose=10)
grid_search.fit(X_train, y_train)

# Test the best model
print("Best parameters:", grid_search.best_params_)
test_score = grid_search.score(X_test, y_test)
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
gen_out_txt = "QKE_GridSearchCV for the " + str(data_name) + " dataset" + "\n"
gen_out_txt = gen_out_txt + "CV-score:" + "\n" + str(grid_search.best_score_) + "\n" + "test-score:" + "\n" + str \
    (test_score) + "\n" + "runtime:" + "\n" + str(end) + "\n"
gen_out_txt = gen_out_txt + "best parameters:" + "\n" + str(grid_search.best_params_) + "\n"
gen_out_txt = gen_out_txt + "GridSearchCV.cv_results:" + "\n" + str(grid_search.cv_results_)

# write CV stats file
InfoPath = "./QKE_GridSearchCV_" + data_name + ".txt"
InfoFile = open(InfoPath, "w+")
InfoFile.write(gen_out_txt)
InfoFile.close()
