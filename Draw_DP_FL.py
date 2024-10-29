import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import csv
import os

# Define the epsilon values you want to plot
epsilon_values = ["None", 0.01, 0.1, 0.5, 1.0, 10.0, 25.0, 50.0]
datasets = ["mnist", "cifar10", "fashion-mnist", "femnist", "shakespeare"]  # List of datasets to plot

# Iterate over each dataset
for dataset_choice in datasets:
    plt.figure(figsize=(10, 6))
    print(f"Plotting results for dataset: {dataset_choice}")
    
    # Iterate over each epsilon value
    for epsilon in epsilon_values:
        # Replace 'None' with 'none' for filename consistency
        epsilon_str = 'none' if epsilon == "None" else str(epsilon)
        csv_filename = f'./log/{dataset_choice}_{epsilon_str}_accuracy_data.csv'
        
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
    plt.title(f'Global Model Accuracy vs Training Rounds for {dataset_choice.capitalize()}')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    save_filename = f'./log/{dataset_choice}_accuracy_comparison.png'
    plt.savefig(save_filename)
    print(f"Plot saved as {save_filename}")
    plt.close()

print("All plots have been generated.")