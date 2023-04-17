import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.neural_networks import TwoLayerQNN
from qiskit_machine_learning.algorithms import NeuralNetworkClassifier
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes

# Load and preprocess the Breast Cancer dataset
breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Define the quantum instance
qi = QuantumInstance(Aer.get_backend('qasm_simulator'), shots=1024)

# Define the feature map and ansatz
feature_map = ZZFeatureMap(feature_dimension=X.shape[1], reps=2)
ansatz = RealAmplitudes(num_qubits=X.shape[1], entanglement='linear')

# Define the quantum neural network and the classifier
qnn = TwoLayerQNN(X.shape[1], feature_map, ansatz, quantum_instance=qi)
classifier = NeuralNetworkClassifier(qnn, loss='cross_entropy')

# Train the classifier
classifier.fit(X_train, y_train)

# Test the classifier
accuracy = classifier.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")