# import os
# import matplotlib.pyplot as plt
# import numpy as np

# def plot_accuracy_vs_rounds(dataset_name):
#     log_directory = './log'
#     epsilon_values = [None, 75.0, 50.0, 25.0, 10.0, 5.0, 1.0, 0.1, 0.01, 0.001]
#     colors = plt.cm.viridis(np.linspace(0, 1, len(epsilon_values))) 

#     # Create a plot
#     plt.figure(figsize=(10, 6))

#     # Check if the directory exists
#     if not os.path.exists(log_directory):
#         print("Log directory does not exist. Ensure you have the right path.")
#         return
    
#     # Loop over each epsilon value and plot its data
#     for epsilon, color in zip(epsilon_values, colors):
#         file_path = f'{log_directory}/{dataset_name}_DP_FedAvg_epsilon_{epsilon}.dat'
#         try:
#             rounds = []
#             accuracies = []
#             with open(file_path, 'r') as file:
#                 for line in file:
#                     round_number, acc = line.strip().split()
#                     rounds.append(int(round_number))
#                     accuracies.append(float(acc))

#             # Plotting data for this epsilon
#             plt.plot(rounds, accuracies, label=f'ε={epsilon}', marker='o', color=color)

#         except FileNotFoundError:
#             print(f"No data file found for {dataset_name} with epsilon {epsilon}. Skipping...")
#             continue

#    # plt.title(f'Accuracy vs. Rounds for Different Epsilon Values ({dataset_name})')
#     plt.xlabel('Rounds', fontsize=16)
#     plt.ylabel('Accuracy (%)', fontsize=16)  
#     plt.legend(title='Epsilon Values', loc='best', fontsize=10.0)  
#     plt.grid(True)
#     plt.tick_params(axis='both', which='major', labelsize=14)
#     plt.savefig(f'./log/{dataset_name}_accuracy_vs_rounds.png')
#     plt.show()

# if __name__ == "__main__":
#     datasets = ['cifar10', 'mnist', 'fashionmnist']
#     for dataset in datasets:
#         plot_accuracy_vs_rounds(dataset)
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_accuracy_vs_rounds(dataset_name, log_directory='./log'):
    epsilon_values = [None, 75.0, 50.0, 25.0, 10.0, 5.0, 1.0, 0.1, 0.01, 0.001]
    colors = plt.cm.viridis(np.linspace(0, 1, len(epsilon_values)))

    plt.figure(figsize=(10, 6))
    for eps, color in zip(epsilon_values, colors):
        eps_str = 'None' if eps is None else str(eps)
        fname = f"{dataset_name}_accuracy_epsilon_{eps_str}.csv"
        fpath = os.path.join(log_directory, fname)
        if not os.path.isfile(fpath):
            print(f"Skipping missing file for ε={eps_str}")
            continue

        df = pd.read_csv(fpath)
        rounds = df.iloc[:, 0].values
        accuracy = df.iloc[:, 1].values

        # plot a point every 10 rounds
        idx = np.arange(0, len(rounds), 10)
        plt.plot(rounds[idx], accuracy[idx],
                 marker='o', label=f'ε={eps_str}', color=color)

    plt.xlabel('Training Rounds')
    plt.ylabel('Model Accuracy (%)')
    plt.title(f'Accuracy vs Rounds for Different ε ({dataset_name.title()})')
    plt.legend(title='Epsilon', loc='upper right')
    plt.grid(True)
    plt.tight_layout()

    out_file = os.path.join(log_directory, f'{dataset_name}_accuracy_vs_rounds.png')
    plt.savefig(out_file)
    plt.close()
    print(f"Saved plot: {out_file}")

if __name__ == "__main__":
    for ds in ['cifar10', 'mnist', 'fashionmnist']:
        plot_accuracy_vs_rounds(ds)