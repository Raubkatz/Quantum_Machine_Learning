import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.neural_networks import TwoLayerQNN
from qiskit_machine_learning.algorithms import NeuralNetworkRegressor
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes

# Load and preprocess the Boston housing dataset
boston = load_boston()
X = boston.data
y = boston.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Define the quantum instance
qi = QuantumInstance(Aer.get_backend('statevector_simulator'))

# Define the feature map and ansatz
feature_map = ZZFeatureMap(feature_dimension=X.shape[1])
ansatz = RealAmplitudes(num_qubits=X.shape[1], entanglement='linear')

# Define the quantum neural network and the regressor
qnn = TwoLayerQNN(X.shape[1], feature_map, ansatz, quantum_instance=qi)
regressor = NeuralNetworkRegressor(qnn, loss='squared_error')

# Train the regressor
regressor.fit(X_train, y_train)

# Test the regressor
r2_score = regressor.score(X_test, y_test)
print(f"R2 score: {r2_score:.2f}")
