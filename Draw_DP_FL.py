import matplotlib.pyplot as plt
import csv
import os

# Define the epsilon values you want to plot
epsilon_values = ["None", 0.01, 0.1, 1.0, 10.0, 25.0, 50.0]
dataset_choice = "cifar10"  # or "cifar10" if you have CIFAR10 data

# Create a plot
plt.figure(figsize=(10, 6))

# Iterate over each epsilon value
for epsilon in epsilon_values:
    csv_filename = f'./log/{dataset_choice}_{epsilon}_accuracy_data.csv'
    
    # Check if the file exists
    if not os.path.exists(csv_filename):
        print(f"File {csv_filename} does not exist. Skipping...")
        continue
    
    rounds = []
    accuracies = []
    
    # Read the CSV file
    with open(csv_filename, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        for row in reader:
            round_num = int(row[0])
            accuracy = float(row[1])
            # Only plot every 10 rounds
            if round_num % 10 == 0:
                rounds.append(round_num)
                accuracies.append(accuracy)
    
    # Plot the data
    plt.plot(rounds, accuracies, marker='o', label=f'ε = {epsilon}')

# Add labels and title
plt.xlabel('Training Rounds')
plt.ylabel('Accuracy (%)')
plt.title('Global Model Accuracy vs Training Rounds')
plt.legend()
plt.grid(True)

# Save the plot
plt.savefig(f'./log/{dataset_choice}_accuracy_comparison.png')

# Show the plot
plt.show()