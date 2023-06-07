import os
import numpy as np

def replace_legend_labels(labels):
    return [label.replace("_simulator", "-simulator").replace("_", ", ") if isinstance(label, str) else label for label in labels]

def replace_table_entries(data):
    return [[entry.replace("_simulator", "-simulator").replace("_", ", ") if isinstance(entry, str) else entry for entry in row] for row in data]

def generate_latex_table(data, header, filename):
    data = replace_table_entries(data)
    with open(filename, 'w') as f:
        f.write('\\begin{tabular}{|' + ' '.join(['c|' for _ in header]) + '}\n')
        f.write('\\hline\n')
        f.write(' & '.join(header) + '\\\\ \\hline\n')

        for row in data:
            f.write(' & '.join([str(x) for x in row]) + '\\\\ \\hline\n')

        f.write('\\end{tabular}\n')

results_path = "results_sun"
files = [f for f in os.listdir(results_path) if f.endswith('.txt')]

merged_accuracies = []
merged_times = []

for file in files:
    kernel_name = file.split('.txt')[0]
    print(kernel_name)
    data = np.loadtxt(os.path.join(results_path, file), skiprows=1)
    sample_sizes = data[:, 0]
    times = data[:, 1]
    accuracies = data[:, 2]

    merged_accuracies.append([kernel_name] + list(accuracies))
    merged_times.append([kernel_name] + list(times))

# Calculate average values for each kernel and add as the final column
merged_accuracies = np.array(sorted(merged_accuracies, key=lambda x: np.mean(x[1:]), reverse=True), dtype=object)
merged_times = np.array(sorted(merged_times, key=lambda x: np.mean(x[1:])), dtype=object)

avg_accuracies = [np.around(np.mean(row[1:]), 6) for row in merged_accuracies]
avg_times = [np.around(np.mean(row[1:]), 6) for row in merged_times]


#merged_accuracies.append(avg_accuracies)
#merged_times.append(avg_times)
from copy import deepcopy as dc
print(np.shape(merged_accuracies))
print(merged_accuracies)

# Create empty arrays with one additional column
accuracies_out = np.empty((len(merged_accuracies), merged_accuracies.shape[1] + 1), dtype=object)
times_out = np.empty((len(merged_times), merged_times.shape[1] + 1), dtype=object)

accuracies_out[:,:-1] = dc(merged_accuracies)
times_out[:,:-1] = dc(merged_times)

#accuracies_out[:,-1] = dc(np.around(np.mean(merged_accuracies[:,1:], axis=1),5))
#times_out[:,-1] = dc(np.around(np.mean(merged_times[:,1:], axis=1),5))

accuracies_out[:,-1] = dc(avg_accuracies)
times_out[:,-1] = dc(avg_times)

# Generate LaTeX tables
header = ["Kernel"] + [f"Size {int(x)}" for x in sample_sizes] + ["Average"]
generate_latex_table(accuracies_out, header, os.path.join(results_path, "accuracies_table_ave.tex"))
generate_latex_table(times_out, header, os.path.join(results_path, "times_table_ave.tex"))