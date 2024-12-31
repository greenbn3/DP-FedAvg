import os
import csv
import matplotlib.pyplot as plt
import numpy as np

# Define the directory containing the CSV files
log_dir = './log'

# Define the datasets (models) and epsilon values to plot
datasets = ["mnist", "cifar10", "fashionmnist"]
epsilon_values = ["5.0", "10.0", "25.0", "50.0", "none"]

# Initialize a dictionary to hold accuracy data
# Structure: {dataset: {epsilon: {round: accuracy}}}
accuracy_data = {dataset: {epsilon: {} for epsilon in epsilon_values} for dataset in datasets}

# Define the range of rounds to consider for averaging
start_round = 0
end_round = 150

# Populate the accuracy_data dictionary
for dataset in datasets:
    for epsilon in epsilon_values:
        csv_filename = os.path.join(log_dir, f"{dataset}_accuracy_epsilon_{epsilon}.csv")
        try:
            with open(csv_filename, mode='r') as file:
                reader = csv.DictReader(file)
                if 'Round' not in reader.fieldnames or 'Accuracy' not in reader.fieldnames:
                    print(f"CSV file '{csv_filename}' is missing required columns. Skipping...")
                    continue
                for row in reader:
                    try:
                        round_num = int(row['Round'])
                        accuracy = float(row['Accuracy'])
                        if start_round <= round_num <= end_round:
                            accuracy_data[dataset][epsilon][round_num] = accuracy
                    except ValueError:
                        print(f"Invalid data in '{csv_filename}': {row}")
                        continue
        except Exception as e:
            print(f"Error reading '{csv_filename}': {e}")

# New graphing function to plot each dataset individually
for dataset in datasets:
    plt.figure()
    for epsilon in epsilon_values:
        rounds = sorted(accuracy_data[dataset][epsilon].keys())
        # Filter rounds to include only those that are multiples of 10
        filtered_rounds = [round for round in rounds if round % 10 == 0]
        accuracies = [accuracy_data[dataset][epsilon][round] for round in filtered_rounds]
        plt.plot(filtered_rounds, accuracies, marker='o', label=f"Epsilon {epsilon}")

    plt.xlabel('Rounds')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy vs Rounds for {dataset}')
    plt.legend()
    plt.xticks(np.arange(start_round, end_round + 1, 10))
    plt.grid(True)
    plt.savefig(f'{dataset}_accuracy.png')
    plt.close()