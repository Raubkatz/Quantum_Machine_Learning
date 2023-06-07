import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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


# Save the external legend
def save_legend(handles, labels, filename, transparent=False, title="Quantum Machine Learning Algorithms",):
    labels = replace_legend_labels(labels)
    fig_legend = plt.figure(figsize=(3, 2))
    ax = fig_legend.add_subplot(111)
    leg = ax.legend(handles, labels, title=title, loc='upper left')

    # Set the background color to white or transparent
    if transparent:
        leg.get_frame().set_facecolor('None')
    else:
        leg.get_frame().set_facecolor('white')

    # Add a frame around the legend
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_linewidth(1.0)

    ax.axis('off')
    fig_legend.savefig(filename, bbox_inches='tight', transparent=transparent)

results_path = "results_sun"
files = [f for f in os.listdir(results_path) if f.endswith('.txt')]

merged_accuracies = []
merged_times = []

# Use seaborn
sns.set()

# Set a color palette with enough colors
color_palette = sns.color_palette("husl", len(files))

# Define line styles
line_styles = ['-', '--', '-.', ':']

for idx, file in enumerate(files):
    kernel_name = file.split('.txt')[0]
    print(kernel_name)
    data = np.loadtxt(os.path.join(results_path, file), skiprows=1)
    sample_sizes = data[:, 0]
    times = data[:, 1]
    accuracies = data[:, 2]

    merged_accuracies.append([kernel_name] + list(accuracies))
    merged_times.append([kernel_name] + list(times))

# Sort merged_accuracies and merged_times by the average accuracy value
merged_accuracies = sorted(merged_accuracies, key=lambda x: np.mean(x[1:]), reverse=True)
merged_times = sorted(merged_times, key=lambda x: x[0])

# Merged plots
plt.figure(figsize=(15, 9))
legend_handles = []
for idx, kernel_data in enumerate(merged_accuracies):
    sns.lineplot(x=sample_sizes, y=kernel_data[1:], marker='o', markersize=8, markeredgewidth=1.5, markeredgecolor='black', linestyle=line_styles[idx % len(line_styles)], color=color_palette[idx], alpha=0.8)
    legend_handles.append(plt.Line2D([0], [0], color=color_palette[idx], lw=2, label=kernel_data[0], linestyle=line_styles[idx % len(line_styles)]))

# Set background color to white
plt.gca().set_facecolor('white')

# Add a frame around the plot area
for spine in plt.gca().spines.values():
    spine.set_visible(True)
    spine.set_color('black')

#plt.title("Merged Accuracies")
plt.xlabel("# of Samples in the Data Set")
plt.ylabel("Accuracy")
plt.grid()
plt.savefig(os.path.join(results_path, "merged_accuracies_sun.png"))
plt.savefig(os.path.join(results_path, "merged_accuracies_sun.eps"))

plt.figure(figsize=(15, 9))
for idx, kernel_data in enumerate(merged_times):
    sns.lineplot(x=sample_sizes, y=kernel_data[1:], marker='o', markersize=8, markeredgewidth=1.5, markeredgecolor='black', linestyle=line_styles[idx % len(line_styles)], color=color_palette[idx], alpha=0.8)

# Set background color to white
plt.gca().set_facecolor('white')

# Add a frame around the plot area
for spine in plt.gca().spines.values():
    spine.set_visible(True)
    spine.set_color('black')

#plt.title("Merged Times")
plt.xlabel("# of Samples in the Data Set")
plt.ylabel("Time (s)")
plt.grid()
plt.savefig(os.path.join(results_path, "merged_times_sun.png"))
plt.savefig(os.path.join(results_path, "merged_times_sun.eps"))


# Save the external legend
save_legend(legend_handles, [kernel_data[0] for kernel_data in merged_accuracies], os.path.join(results_path, "external_legend_sun.png"))
save_legend(legend_handles, [kernel_data[0] for kernel_data in merged_accuracies], os.path.join(results_path, "external_legend_sun.eps"))


# Sort merged_times by the average runtime value
merged_times = sorted(merged_times, key=lambda x: np.mean(x[1:]))

# Generate LaTeX tables
header = ["Kernel"] + [f"Size {int(x)}" for x in sample_sizes]
generate_latex_table(merged_accuracies, header, os.path.join(results_path, "accuracies_table_sun.tex"))
generate_latex_table(merged_times, header, os.path.join(results_path, "times_table_sun.tex"))
