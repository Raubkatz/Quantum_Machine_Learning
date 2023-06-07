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
from qiskit.circuit.library import PauliFeatureMap, ZFeatureMap, ZZFeatureMap, EfficientSU2, TwoLocal, RealAmplitudes
from qiskit.algorithms.optimizers import COBYLA, SPSA, NFT
from qiskit.providers.aer import Aer
from qiskit.utils import QuantumInstance
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
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import time
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# Create results folder if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')




#sample sizes:
#sample_sizes = np.arange(40, 61, 10) #test
sample_sizes = [50,100,250,500,1000,1500,2000]
random_combinations = 20
random.seed(7)

use_PCA = "No"
pca_prefix = "PCA_" if use_PCA == "Yes" else ""

feature_maps = [ZFeatureMap, PauliFeatureMap, ZZFeatureMap]
quantum_instances = [
    QuantumInstance(Aer.get_backend('aer_simulator'), shots=1024),
    QuantumInstance(Aer.get_backend('qasm_simulator'), shots=1024),
    QuantumInstance(Aer.get_backend('statevector_simulator'), shots=1024)
]
C_values = np.logspace(-3, 3, 9)

quantum_kernels = [
    {
        "name": f"{fm.__name__}_{qi.backend_name}_{C}",
        "feature_map": fm(feature_dimension=2),  # Set feature_dimension to 2 since it's necessary for creating feature_map instances
        "backend": qi.backend,
        "C": C
    }
    for fm in feature_maps
    for qi in quantum_instances
    for C in C_values
]

did_not_work_list = list()

jjj=0

print('Number of parametrizations:')
print(len(quantum_kernels))
print('')

random_quantum_kernels = random.sample(quantum_kernels, random_combinations)

for kernel in random_quantum_kernels:
    name = kernel['name']
    feature_map = kernel['feature_map']
    quantum_instance = QuantumInstance(kernel['backend'], shots=1024, seed_simulator=42, seed_transpiler=42)
    C = kernel['C']  # Extract C from the kernel dictionary


    times = []
    accuracies = []

    try:
        for sample_size in sample_sizes:
            print(f"Classifying with {sample_size} samples using {name}")
            print(jjj)
            jjj = jjj + 1
            # Generate artificial dataset
            X, y = make_classification(n_samples=sample_size, n_features=2, n_informative=2, n_redundant=0,
                                       n_clusters_per_class=1, random_state=42)

            if use_PCA == "Yes":
                # Perform PCA on the generated data
                pca = PCA(n_components=2)
                X = pca.fit_transform(X)

            # Normalize and rescale the data
            scaler = MinMaxScaler(feature_range=(0, 1))
            X = scaler.fit_transform(X)

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Create the QuantumKernel object
            start_time = time.time()
            print('Precalculating the quantum kernel')
            quantum_kernel = QuantumKernel(feature_map=feature_map, quantum_instance=quantum_instance)

            # Train and evaluate the updated Quantum Kernel Estimator
            qk_matrix_train = quantum_kernel.evaluate(x_vec=X_train)
            qk_matrix_test = quantum_kernel.evaluate(x_vec=X_test, y_vec=X_train)
            print('Quantum Kernel Classification')
            svc = SVC(kernel='precomputed', C=C).fit(qk_matrix_train, y_train)
            y_pred_svc = svc.predict(qk_matrix_test)
            end_time = time.time()

            # Calculate the time taken
            time_taken = end_time - start_time
            times.append(time_taken)

            # Calculate the accuracy
            accuracy_svc = accuracy_score(y_test, y_pred_svc)
            accuracies.append(accuracy_svc)

            print(f"Time taken for {sample_size} samples: {time_taken:.2f} seconds")
            print(f"Accuracy for {sample_size} samples: {accuracy_svc:.2f}\n")

        # Save all iterations in a single text file
        results_array = np.column_stack((sample_sizes, times, accuracies))
        np.savetxt(f"results/{pca_prefix}QKE_results_{name}.txt", results_array, fmt='%.6f',
                   header='Sample_Size Time(s) Accuracy', delimiter='\t', comments='')

    except:
        did_not_work_string = str(name) + "\n" + str(feature_map) + "\n" + str(quantum_instance) + "\n" + str(C) + "\n\n##########################################\n\n"
        did_not_work_list.append(did_not_work_string)
        print(did_not_work_string)
        print('did not work')

did_not_work_list.append('This is the end.')
print('Saving failed parametrizations')
np.savetxt("QKE_failed_parametrizations.txt", did_not_work_list, fmt='%s')
