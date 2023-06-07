import os
import numpy as np
import matplotlib.pyplot as plt

def generate_latex_table(data, header, filename):
    with open(filename, 'w') as f:
        f.write('\\begin{tabular}{|' + ' '.join(['c|' for _ in header]) + '}\n')
        f.write('\\hline\n')
        f.write(' & '.join(header) + '\\\\ \\hline\n')

        for row in data:
            f.write(' & '.join([str(x) for x in row]) + '\\\\ \\hline\n')

        f.write('\\end{tabular}\n')


def save_legend(handles, labels, filename):
    fig = plt.figure()
    plt.legend(handles, labels)
    plt.axis('off')
    fig.canvas.draw()
    plt.savefig(filename, format='png')


results_path = "results"
files = [f for f in os.listdir(results_path) if f.endswith('.txt')]

merged_accuracies = []
merged_times = []

plot_size = (10, 7)  # Increase the plot size (width, height)

for file in files:
    kernel_name = file.split('.')[0]
    print(kernel_name)
    data = np.loadtxt(os.path.join(results_path, file), skiprows=1)
    sample_sizes = data[:, 0]
    times = data[:, 1]
    accuracies = data[:, 2]

    merged_accuracies.append([kernel_name] + list(accuracies))
    merged_times.append([kernel_name] + list(times))

    # Individual kernel plots
    #plt.figure(figsize=plot_size)
    #plt.plot(sample_sizes, accuracies, marker='o')
    #plt.title(f"{kernel_name} - Accuracy")
    #plt.xlabel("Sample Size")
    #plt.ylabel("Accuracy")
    #plt.grid()
    #plt.savefig(os.path.join(results_path, f"{kernel_name}_accuracy.png"))
    #plt.savefig(os.path.join(results_path, f"{kernel_name}_accuracy.eps"), format='eps')

    #plt.figure(figsize=plot_size)
    #plt.plot(sample_sizes, times, marker='o')
    #plt.title(f"{kernel_name} - Time")
    #plt.xlabel("Sample Size")
    #plt.ylabel("Time (s)")
    #plt.grid()
    #plt.savefig(os.path.join(results_path, f"{kernel_name}_time.png"))
    #plt.savefig(os.path.join(results_path, f"{kernel_name}_time.eps"), format='eps')

# Merged plots
legend_handles = []

plt.figure(figsize=plot_size)
for kernel_data in merged_accuracies:
    handle, = plt.plot(sample_sizes, kernel_data[1:], marker='o', label=kernel_data[0])
    legend_handles.append(handle)
plt.title("Merged Accuracies")
plt.xlabel("Sample Size")
plt.ylabel("Accuracy")
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(results_path, "merged_accuracies.png"))
plt.savefig(os.path.join(results_path, "merged_accuracies.eps"), format='eps')

save_legend(legend_handles, [kd[0] for kd in merged_accuracies], os.path.join(results_path, "legend.png"))

legend_handles.clear()

plt.figure(figsize=plot_size)
for kernel_data in merged_times:
    handle, = plt.plot(sample_sizes, kernel_data[1:], marker='o', label=kernel_data[0])
    legend_handles.append(handle)
plt.title("Merged Times")
plt.xlabel("Sample Size")
plt.ylabel("Time (s)")
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(results_path, "merged_times.png"))
plt.savefig(os.path.join(results_path, "merged_times.eps"), format='eps')

save_legend(legend_handles, [kd[0] for kd in merged_times], os.path.join(results_path, "legend_times.png"))

# Generate LaTeX tables
header = ["Kernel"] + [f"Size {int(x)}" for x in sample_sizes]
generate_latex_table(merged_accuracies, header, os.path.join(results_path, "accuracies_table.tex"))
generate_latex_table(merged_times, header, os.path.join(results_path, "times_table.tex"))

