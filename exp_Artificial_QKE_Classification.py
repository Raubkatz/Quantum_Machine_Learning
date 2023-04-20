import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
import time
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit import BasicAer
from qiskit.utils import QuantumInstance

from qiskit import Aer
from qiskit.circuit.library import ZFeatureMap, PauliFeatureMap, ZZFeatureMap, RealAmplitudes
import os

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from qiskit import Aer, BasicAer
from qiskit.circuit.library import ZFeatureMap, PauliFeatureMap, ZZFeatureMap
import time
from qiskit import Aer, BasicAer
from qiskit.circuit.library import ZFeatureMap, PauliFeatureMap, ZZFeatureMap
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit.utils import QuantumInstance
import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# Create results folder if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')

sample_sizes = np.arange(500, 2001, 500)

# Set PCA to "Yes" or "No"
use_PCA = "No"
pca_prefix = "PCA_" if use_PCA == "Yes" else ""

###already calculated with 500-2000 samples
'''
    {
        "name": "ZFeatureMap_reps_2",
        "feature_map": ZFeatureMap(feature_dimension=2, reps=2),
        "backend": BasicAer.get_backend('qasm_simulator')
    },
    {
        "name": "PauliFeatureMap_Z_ZZ_reps_2",
        "feature_map": PauliFeatureMap(feature_dimension=2, reps=2, paulis=['Z', 'ZZ']),
        "backend": Aer.get_backend('statevector_simulator')
    },
    {
        "name": "PauliFeatureMap_X_Y_Z_reps_1",
        "feature_map": PauliFeatureMap(feature_dimension=2, reps=1, paulis=['X', 'Y', 'Z']),
        "backend": Aer.get_backend('statevector_simulator')
    },
    {
        "name": "ZZFeatureMap_linear_reps_1",
        "feature_map": ZZFeatureMap(feature_dimension=2, reps=1, entanglement='linear'),
        "backend": BasicAer.get_backend('qasm_simulator')
    },
    
    {
        "name": "PauliFeatureMap_X_Y_reps_2",
        "feature_map": PauliFeatureMap(feature_dimension=4, reps=2, paulis=['X', 'Y']),
        "backend": Aer.get_backend('statevector_simulator')
    },
    '''

# Define quantum kernels
quantum_kernels = [
    {
        "name": "PauliFeatureMap_X_Y_reps_2",
        "feature_map": PauliFeatureMap(feature_dimension=4, reps=2, paulis=['X', 'Y']),
        "backend": Aer.get_backend('statevector_simulator')
    },
    {
        "name": "ZFeatureMap_reps_3",
        "feature_map": ZFeatureMap(feature_dimension=2, reps=3),
        "backend": BasicAer.get_backend('qasm_simulator')
    },
    {
        "name": "PauliFeatureMap_X_Y_Z_reps_1_v2",
        "feature_map": PauliFeatureMap(feature_dimension=2, reps=1, paulis=['X', 'Y', 'Z']),
        "backend": Aer.get_backend('statevector_simulator')
    },
    {
        "name": "ZZFeatureMap_full_reps_2",
        "feature_map": ZZFeatureMap(feature_dimension=2, reps=2, entanglement='full'),
        "backend": Aer.get_backend('statevector_simulator')
    }
]

for kernel in quantum_kernels:
    kernel_name = kernel["name"]
    feature_map = kernel["feature_map"]
    backend = kernel["backend"]
    quantum_instance = QuantumInstance(backend, shots=1024, seed_simulator=42, seed_transpiler=42)
    quantum_kernel = QuantumKernel(feature_map=feature_map, quantum_instance=quantum_instance)

    times = []
    accuracies = []

    for sample_size in sample_sizes:
        print(f"Classifying with {sample_size} samples using {kernel_name}")

        # Generate artificial dataset
        X, y = make_classification(n_samples=sample_size, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=42)

        if use_PCA == "Yes":
            # Perform PCA on the generated data
            pca = PCA(n_components=2)
            X = pca.fit_transform(X)

        # Normalize and rescale the data
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X = scaler.fit_transform(X)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train and evaluate the Quantum Kernel SVC model
        start_time = time.time()
        qk_matrix_train = quantum_kernel.evaluate(x_vec=X_train)
        qk_matrix_test = quantum_kernel.evaluate(x_vec=X_test, y_vec=X_train)
        svc = SVC(kernel='precomputed').fit(qk_matrix_train, y_train)
        y_pred_svc = svc.predict(qk_matrix_test)
        end_time = time.time()

        # Calculate the time taken
        time_taken = end_time - start_time
        times.append(time_taken)

        # Calculate the accuracy
        accuracy = svc.score(qk_matrix_test, y_test)
        accuracies.append(accuracy)

        print(f"Time taken for {sample_size} samples: {time_taken:.2f} seconds")
        print(f"Accuracy for {sample_size} samples: {accuracy:.2f}\n")

    # Save all iterations in a single text file
    results_array = np.column_stack((sample_sizes, times, accuracies))
    np.savetxt(f"results/{pca_prefix}QKE_SVM_results_{kernel_name}.txt", results_array, fmt='%.6f',
               header='Sample_Size Time(s) Accuracy', delimiter='\t', comments='')

    # Plot the graph comparing runtimes of all iterations
    plt.plot(sample_sizes, times)
    plt.xlabel('Number of Samples')
    plt.ylabel('Runtime (seconds)')
    plt.title(f'Runtime vs Number of Samples for Quantum Kernel SVC ({kernel_name})')
    plt.grid()
    plt.savefig(f'results/{pca_prefix}runtime_plot_{kernel_name}.png')
    plt.show()

    # Plot the graph comparing accuracies of all iterations
    plt.plot(sample_sizes, accuracies)
    plt.xlabel('Number of Samples')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy vs Number of Samples for Quantum Kernel SVC ({kernel_name})')
    plt.grid()
    plt.savefig(f'results/{pca_prefix}accuracy_plot_{kernel_name}.png')
    plt.show()


'''
# Create results folder if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')

sample_sizes = np.arange(500, 2001, 500)
times = []
accuracies = []

for sample_size in sample_sizes:
    print(f"Classifying with {sample_size} samples")

    # Generate artificial dataset
    X, y = make_classification(n_samples=sample_size, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=42)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a custom quantum kernel
    feature_map = PauliFeatureMap(feature_dimension=X.shape[1], reps=1, paulis=['X', 'Y', 'Z'])
    backend = Aer.get_backend('statevector_simulator')
    quantum_instance = QuantumInstance(backend, shots=1024)
    quantum_kernel = QuantumKernel(feature_map=feature_map, quantum_instance=quantum_instance)

    # Train and evaluate the Quantum Kernel SVC model
    start_time = time.time()
    qk_matrix_train = quantum_kernel.evaluate(x_vec=X_train)
    qk_matrix_test = quantum_kernel.evaluate(x_vec=X_test, y_vec=X_train)
    svc = SVC(kernel='precomputed').fit(qk_matrix_train, y_train)
    y_pred_svc = svc.predict(qk_matrix_test)
    end_time = time.time()

    # Calculate the time taken
    time_taken = end_time - start_time
    times.append(time_taken)

    # Calculate the accuracy
    accuracy = svc.score(qk_matrix_test, y_test)
    accuracies.append(accuracy)

    print(f"Time taken for {sample_size} samples: {time_taken:.2f} seconds")
    print(f"Accuracy for {sample_size} samples: {accuracy:.2f}\n")

# Save all iterations in a single text file
results_array = np.column_stack((sample_sizes, times, accuracies))
np.savetxt("results/QKE_SVM_results.txt", results_array, fmt='%.6f', header='Sample_Size Time(s) Accuracy', delimiter='\t', comments='')

# Plot the graph comparing runtimes of all iterations
plt.plot(sample_sizes, times)
plt.xlabel('Number of Samples')
plt.ylabel('Runtime (seconds)')
plt.title('Runtime vs Number of Samples for Quantum Kernel SVC')
plt.grid()
plt.savefig('results/runtime_plot.png')
plt.show()

# Plot the graph comparing accuracies of all iterations
plt.plot(sample_sizes, accuracies)
plt.xlabel('Number of Samples')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Number of Samples for Quantum Kernel SVC')
plt.grid()
plt.savefig('results/accuracy_plot.png')
plt.show()

--------------
---------------
sample_sizes = np.arange(500, 2001, 500)
times = []

for sample_size in sample_sizes:
    print(f"Classifying with {sample_size} samples")

    # Generate artificial dataset
    X, y = make_classification(n_samples=sample_size, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=42)

    # Split data into training and testing sets
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
    quantum_kernel = quantum_kernel_3


    # Train and evaluate the Quantum Kernel SVC model
    start_time = time.time()
    qk_matrix_train = quantum_kernel.evaluate(x_vec=X_train)
    qk_matrix_test = quantum_kernel.evaluate(x_vec=X_test, y_vec=X_train)
    svc = SVC(kernel='precomputed').fit(qk_matrix_train, y_train)
    y_pred_svc = svc.predict(qk_matrix_test)
    end_time = time.time()

    # Calculate the time taken
    time_taken = end_time - start_time
    times.append(time_taken)

    print(f"Time taken for {sample_size} samples: {time_taken:.2f} seconds\n")

# Plot the graph comparing runtimes of all iterations
plt.plot(sample_sizes, times)
plt.xlabel('Number of Samples')
plt.ylabel('Runtime (seconds)')
plt.title('Runtime vs Number of Samples for Quantum Kernel SVC')
plt.grid()
plt.show()
'''