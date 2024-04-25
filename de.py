import os
import matplotlib.pyplot as plt
import numpy as np

def plot_accuracy_vs_rounds(dataset_name):
    log_directory = './log'
    epsilon_values = [0.002, 1.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(epsilon_values)))  # Generate a color map

    # Create a plot
    plt.figure(figsize=(10, 6))

    # Check if the directory exists
    if not os.path.exists(log_directory):
        print("Log directory does not exist. Ensure you have the right path.")
        return

    # Loop over each epsilon value and plot its data
    plotted = False
    for epsilon, color in zip(epsilon_values, colors):
        file_path = f'{log_directory}/{dataset_name}_DP_FedAvg_epsilon_{epsilon}.dat'
        try:
            rounds = []
            accuracies = []
            with open(file_path, 'r') as file:
                for line in file:
                    round_number, acc = line.strip().split()
                    rounds.append(int(round_number))
                    accuracies.append(float(acc))
            # Plotting data for this epsilon
            plt.plot(rounds, accuracies, label=f'ε={epsilon}', marker='o', color=color)
            plotted = True
        except FileNotFoundError:
            print(f"No data file found for {dataset_name} with epsilon {epsilon}. Skipping...")
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")

    if not plotted:
        print(f"No data plotted for {dataset_name}. Check data files and content.")
    else:
        plt.title(f'Accuracy vs. Rounds for Different Epsilon Values ({dataset_name})')
        plt.xlabel('Rounds')
        plt.ylabel('Accuracy (%)')
        plt.legend(title='Epsilon Values', loc='best')
        plt.grid(True)
        plt.savefig(f'./log/{dataset_name}_accuracy_vs_rounds.png')
        plt.show()

if __name__ == "__main__":
    datasets = ['CIFAR10', 'MNIST']
    for dataset in datasets:
        plot_accuracy_vs_rounds(dataset)

