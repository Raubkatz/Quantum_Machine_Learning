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
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import random

# Create results folder if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')

#sample sizes:
#sample_sizes = np.arange(40, 61, 10) #test only
sample_sizes = [50,100,250,500,1000,1500,2000]
random_combinations = 20
random.seed(42)
max_iter = 100 #maximum iterations for the optimizer

# Set PCA to "Yes" or "No"
use_PCA = "No"
pca_prefix = "PCA_" if use_PCA == "Yes" else ""

# Define VQC parameters
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
        QuantumInstance(Aer.get_backend('statevector_simulator'), shots=1024)
    ],
}

jjj = 0

VQC_parametrizations = [
    {
        'name': f"{fm.__name__}_{ansatz.__name__}_{optimizer.__class__.__name__}_{qi.backend_name}",
        'feature_map': fm,
        'ansatz': ansatz,
        'optimizer': optimizer,
        'quantum_instance': qi,
    }
    for fm in param_grid['feature_map']
    for ansatz in param_grid['ansatz']
    for optimizer in param_grid['optimizer']
    for qi in param_grid['quantum_instance']
]

# Replace the previous loop header with this:
for vqc_param in VQC_parametrizations:
    name = vqc_param["name"]
    feature_map = vqc_param["feature_map"]
    ansatz = vqc_param["ansatz"]
    optimizer = vqc_param["optimizer"]
    quantum_instance = vqc_param["quantum_instance"]

random_params = random.sample(VQC_parametrizations, random_combinations)

did_not_work_list = list()

for params in random_params:
    name = params['name']
    feature_map = params['feature_map']
    ansatz = params['ansatz']
    optimizer = params['optimizer']
    quantum_instance = params['quantum_instance']

    times = []
    accuracies = []

    try:
        for sample_size in sample_sizes:
            print(jjj)
            jjj = jjj + 1
            print(f"Classifying with {sample_size} samples using {name}")

            # Generate artificial dataset
            X, y = make_classification(n_samples=sample_size, n_features=2, n_informative=2, n_redundant=0,
                                       n_clusters_per_class=1, random_state=42)

            if use_PCA == "Yes":
                # Perform PCA on the generated data
                pca = PCA(n_components=20)
                X = pca.fit_transform(X)

            # Normalize and rescale the data
            scaler = MinMaxScaler(feature_range=(0, 1))
            X = scaler.fit_transform(X)

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

            ## Binarize labels
            #label_binarizer = LabelBinarizer()
            #y_train_onehot = label_binarizer.fit_transform(y_train)
            #y_test_onehot = label_binarizer.transform(y_test)

            # Create the VQC model
            vqc = VQC(feature_map=feature_map(X.shape[1]),
                      ansatz=ansatz(num_qubits=X.shape[1]),
                      optimizer=optimizer,
                      quantum_instance=quantum_instance)

            # Train and evaluate the VQC model
            start_time = time.time()
            vqc.fit(X_train, y_train_onehot)
            y_pred_vqc = vqc.predict(X_test)
            end_time = time.time()

            # Calculate the time taken
            time_taken = end_time - start_time
            times.append(time_taken)

            # Calculate the accuracy
            accuracy_vqc = accuracy_score(y_test_onehot, y_pred_vqc)
            accuracies.append(accuracy_vqc)

            print(f"Time taken for {sample_size} samples: {time_taken:.2f} seconds")
            print(f"Accuracy for {sample_size} samples: {accuracy_vqc:.2f}\n")

        # Save all iterations in a single text file
        results_array = np.column_stack((sample_sizes, times, accuracies))
        np.savetxt(f"results/{pca_prefix}VQC_results_{name}.txt", results_array, fmt='%.6f',
                   header='Sample_Size Time(s) Accuracy', delimiter='\t', comments='')

    except:
        did_not_work_string = str(name) + "\n" + str(feature_map) + "\n" + str(ansatz) + "\n" + str(optimizer) + "\n" + str(quantum_instance) + "\n\n##########################################\n\n"
        did_not_work_list.append(did_not_work_string)
        print(did_not_work_string)
        print('did not work')

did_not_work_list.append('This is the end.')
print('Saving failed paramtrizations')
np.savetxt("VQC_failed_parametrizations.txt", did_not_work_list, fmt='%s')