import os
import csv
import matplotlib.pyplot as plt
import numpy as np

# Define the directory containing the CSV files
log_dir = './log'

# Define the datasets (models) and epsilon values to plot
datasets = ["femnist", "fashion-mnist", "shakespeare", "mnist", "cifar10"]
epsilon_values = ["0.01", "0.1", "0.5", "1.0", "10.0", "25.0", "50.0", "none"]

# Initialize a dictionary to hold accuracy data
# Structure: {dataset: {epsilon: average_accuracy}}
accuracy_data = {dataset: {} for dataset in datasets}

# Define the range of rounds to consider for averaging
start_round = 40
end_round = 100

# Populate the accuracy_data dictionary
for dataset in datasets:
    for epsilon in epsilon_values:
        # Handle 'none' epsilon
        epsilon_str = 'none' if epsilon.lower() == "none" else epsilon

        # Construct the CSV file name
        csv_filename = os.path.join(log_dir, f'{dataset}_{epsilon_str}_accuracy_data.csv')

        # Check if the CSV file exists
        if not os.path.exists(csv_filename):
            print(f"CSV file '{csv_filename}' does not exist. Skipping...")
            continue

        # Read the CSV file and collect accuracies for rounds 40-100
        try:
            with open(csv_filename, mode='r') as file:
                reader = csv.DictReader(file)
                # Check if required columns exist
                if 'Round' not in reader.fieldnames or 'Accuracy' not in reader.fieldnames:
                    print(f"CSV file '{csv_filename}' is missing required columns. Skipping...")
                    continue
                accuracies = []
                for row in reader:
                    try:
                        round_num = int(row['Round'])
                        accuracy = float(row['Accuracy'])
                        if start_round <= round_num <= end_round:
                            accuracies.append(accuracy)
                    except ValueError:
                        print(f"Invalid data in '{csv_filename}': {row}")
                        continue

                if accuracies:
                    average_accuracy = sum(accuracies) / len(accuracies)
                    accuracy_data[dataset][epsilon] = average_accuracy
                else:
                    print(f"No accuracy data found in rounds {start_round}-{end_round} in '{csv_filename}'.")
        except Exception as e:
            print(f"Error reading '{csv_filename}': {e}")

# Prepare the plot
plt.figure(figsize=(14, 8))

# Define colors for each dataset
color_map = {
    "femnist": "blue",
    "fashion-mnist": "green",
    "shakespeare": "red",
    "mnist": "purple",
    "cifar10": "orange",
    # Add more datasets here if needed
}

# Assign each epsilon value a unique integer position for equal spacing
# e.g., "0.01" -> 0, "0.1" -> 1, ..., "none" -> 7
epsilon_positions = {epsilon: idx for idx, epsilon in enumerate(epsilon_values)}
# Create labels and positions for the x-axis
x_labels = epsilon_values
x_positions = list(range(len(epsilon_values)))

# Iterate over each dataset to plot
for dataset in datasets:
    epsilons = []
    averages = []
    for epsilon in epsilon_values:
        epsilon_str = 'none' if epsilon.lower() == "none" else epsilon
        avg_acc = accuracy_data[dataset].get(epsilon)
        if avg_acc is not None:
            pos = epsilon_positions[epsilon]
            epsilons.append(pos)
            averages.append(avg_acc)

    if epsilons and averages:
        plt.plot(
            epsilons,
            averages,
            marker='o',
            label=dataset,
            color=color_map.get(dataset, None)
        )

# Customize the x-axis
plt.xticks(x_positions, x_labels)
plt.xlabel('Epsilon Values')
plt.ylabel('Average Accuracy (%) (Rounds 40-100)')
plt.title('Model Average Accuracy vs. Epsilon Values')
plt.legend(title='Datasets')
plt.grid(True, linestyle='--', linewidth=0.5)

# Adjust layout for better spacing
plt.tight_layout()

# Save the plot
plot_path = os.path.join(log_dir, 'average_accuracy_vs_epsilon_equal_spacing.png')
plt.savefig(plot_path)
print(f"Plot saved as 'average_accuracy_vs_epsilon_equal_spacing.png' in the '{log_dir}' directory.")

# Display the plot
plt.show()

