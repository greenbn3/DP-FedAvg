import os
import matplotlib.pyplot as plt

def plot_accuracy_vs_epsilon():
    log_directory = './log'
    epsilon_values = [0.002, 1.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    accuracies = []
    
    # Check if the directory exists
    if not os.path.exists(log_directory):
        print("Log directory does not exist. Ensure you have the right path.")
        return
    
    # Read the fifth line (round 5) from each accuracy file
    for epsilon in epsilon_values:
        file_path = f'{log_directory}/accuracy_epsilon_{epsilon}.dat'
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
                round_number, acc = lines[4].strip().split()  # Get the 5th line (index 4), split it
                accuracies.append(float(acc))  # Append only the accuracy, converted to float
        except FileNotFoundError:
            print(f"No data file found for epsilon {epsilon}. Skipping...")
            continue
        except IndexError:
            print(f"Not enough data in file for epsilon {epsilon}. Expected at least 5 rounds.")
            continue

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(epsilon_values, accuracies, marker='o', linestyle='-', color='b')
    plt.title('Accuracy vs. Epsilon at Round 5')
    plt.xlabel('Epsilon')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    plot_accuracy_vs_epsilon()

