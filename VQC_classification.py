import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelBinarizer
from sklearn.metrics import accuracy_score
from qiskit import BasicAer
from qiskit import Aer

from qiskit.utils import QuantumInstance
from qiskit.circuit.library import ZFeatureMap, PauliFeatureMap, ZZFeatureMap, RealAmplitudes, TwoLocal
from qiskit_machine_learning.algorithms import VQC
from qiskit.opflow import Z
from qiskit.algorithms.optimizers import COBYLA, SPSA, ADAM, L_BFGS_B, TNC, NFT
from qiskit.circuit.library import EfficientSU2

# Load and preprocess the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Standardize the features
scaler = MinMaxScaler().fit(X)
X = scaler.transform(X)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Binarize labels
label_binarizer = LabelBinarizer()
y_train_onehot = label_binarizer.fit_transform(y_train)
y_test_onehot = label_binarizer.transform(y_test)

# Define the feature map
feature_map = ZZFeatureMap(feature_dimension=X.shape[1], reps=2, entanglement='linear')

# Define the optimizer
optimizer_1 = COBYLA(maxiter=100)
optimizer_2 = SPSA(maxiter=100)
optimizer_3 = ADAM(maxiter=100)
optimizer_4 = L_BFGS_B(maxiter=100)
optimizer_5 = TNC(maxiter=100)
optimizer_6 = NFT(maxiter=100)
optimizer = optimizer_2


# Define the backend
backend = Aer.get_backend('statevector_simulator')
quantum_instance = QuantumInstance(backend, shots=1024)

# Define the variational circuits
var_circuit_1 = EfficientSU2(num_qubits=X.shape[1], reps=2, entanglement='linear', insert_barriers=True)
var_circuit_2 = RealAmplitudes(num_qubits=X.shape[1], reps=2, entanglement='linear', insert_barriers=True)
var_circuit_3 = TwoLocal(num_qubits=X.shape[1], reps=2, rotation_blocks=['ry', 'rz'], entanglement_blocks='cx', entanglement='linear', insert_barriers=True)

# Choose which variational circuit to use
var_circuit = var_circuit_1

# Create the VQC model
vqc = VQC(feature_map=feature_map, ansatz=var_circuit, optimizer=optimizer, quantum_instance=quantum_instance)

# Train and evaluate the VQC model
vqc.fit(X_train, y_train_onehot)
y_pred_vqc = vqc.predict(X_test)
accuracy_vqc = accuracy_score(y_test_onehot, y_pred_vqc)

print(f'VQC accuracy: {accuracy_vqc}')
