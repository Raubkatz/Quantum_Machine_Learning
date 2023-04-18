import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score
from qiskit import BasicAer
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit import Aer
from qiskit.circuit.library import ZFeatureMap, PauliFeatureMap, ZZFeatureMap, RealAmplitudes
from sklearn.svm import SVC #Very important, we're going to use support vector machines from sklearn and plug in our qunatum kernels
from sklearn.metrics import accuracy_score, classification_report

# Load and preprocess the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Load and preprocess the Breast Cancer dataset
#breast_cancer = datasets.load_breast_cancer()
#X = breast_cancer.data
#y = breast_cancer.target

# Load and preprocess the Heart Disease dataset
#heart = datasets.load_heart_disease()
#X = heart.data
#y = heart.target

# Standardize the features
#scaler = StandardScaler().fit(X)
scaler = MinMaxScaler().fit(X)
X = scaler.transform(X)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a custom quantum kernel #Qunatum Kernel #1
feature_map_1 = ZFeatureMap(feature_dimension=X.shape[1], reps=2)
backend_1 = BasicAer.get_backend('qasm_simulator')
quantum_instance_1 = QuantumInstance(backend_1, shots=1024, seed_simulator=42, seed_transpiler=42)
quantum_kernel_1 = QuantumKernel(feature_map=feature_map_1, quantum_instance=quantum_instance_1)

# Create a custom quantum kernel #Qunatum Kernel #2
feature_map_2 = PauliFeatureMap(feature_dimension=X.shape[1], reps=2, paulis=['Z', 'ZZ'])
backend_2 = Aer.get_backend('statevector_simulator')
quantum_instance_2 = QuantumInstance(backend_2, shots=1024)
quantum_kernel_2 = QuantumKernel(feature_map=feature_map_2, quantum_instance=quantum_instance_2)

# Create a custom quantum kernel - Quantum Kernel Option #3
feature_map_3 = PauliFeatureMap(feature_dimension=X.shape[1], reps=1, paulis=['X', 'Y', 'Z'])
backend_3 = Aer.get_backend('statevector_simulator')
quantum_instance_3 = QuantumInstance(backend_3, shots=1024)
quantum_kernel_3 = QuantumKernel(feature_map=feature_map_3, quantum_instance=quantum_instance_3)

# Create a custom quantum kernel - Quantum Kernel Option #4
feature_map_4 = ZZFeatureMap(feature_dimension=X.shape[1], reps=1, entanglement='linear')
backend_4 = BasicAer.get_backend('qasm_simulator')
quantum_instance_4 = QuantumInstance(backend_4, shots=1024, seed_simulator=42, seed_transpiler=42)
quantum_kernel_4 = QuantumKernel(feature_map=feature_map_4, quantum_instance=quantum_instance_4)

# Create a custom quantum kernel - Quantum Kernel Option #5
feature_map_5 = PauliFeatureMap(feature_dimension=4, reps=2, paulis=['X', 'Y'])
backend_5 = Aer.get_backend('statevector_simulator')
quantum_instance_5 = QuantumInstance(backend_5, shots=1024)
quantum_kernel_5 = QuantumKernel(feature_map=feature_map_5, quantum_instance=quantum_instance_5)

# Create a custom quantum kernel - Quantum Kernel Option #7
feature_map_6 = ZFeatureMap(feature_dimension=X.shape[1], reps=3)
backend_6 = BasicAer.get_backend('qasm_simulator')
quantum_instance_6 = QuantumInstance(backend_6, shots=1024, seed_simulator=42, seed_transpiler=42)
quantum_kernel_6 = QuantumKernel(feature_map=feature_map_6, quantum_instance=quantum_instance_6)

# Create a custom quantum kernel - Quantum Kernel Option #8
feature_map_7 = PauliFeatureMap(feature_dimension=X.shape[1], reps=1, paulis=['X', 'Y', 'Z'])
backend_7 = Aer.get_backend('statevector_simulator')
quantum_instance_7 = QuantumInstance(backend_7, shots=1024)
quantum_kernel_7 = QuantumKernel(feature_map=feature_map_7, quantum_instance=quantum_instance_7)

# Create a custom quantum kernel - Quantum Kernel Option #9
feature_map_8 = ZZFeatureMap(feature_dimension=X.shape[1], reps=2, entanglement='full')
backend_8 = Aer.get_backend('statevector_simulator')
quantum_instance_8 = QuantumInstance(backend_8, shots=1024)
quantum_kernel_8 = QuantumKernel(feature_map=feature_map_8, quantum_instance=quantum_instance_8)

# Choose which quantum kernel to use
quantum_kernel = quantum_kernel_4

# Train and evaluate the Quantum Kernel SVC model
qk_matrix_train = quantum_kernel.evaluate(x_vec=X_train)
qk_matrix_test = quantum_kernel.evaluate(x_vec=X_test, y_vec=X_train)
svc = SVC(kernel='precomputed').fit(qk_matrix_train, y_train)
y_pred_svc = svc.predict(qk_matrix_test)
accuracy_svc = accuracy_score(y_test, y_pred_svc)

print(f'SVC with Quantum Kernel accuracy: {accuracy_svc}')

report = classification_report(y_test, y_pred_svc, target_names=iris.target_names)
print("Classification Report:\n", report)

