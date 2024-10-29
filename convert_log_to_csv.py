import os
import csv

# Define the directory containing the log files
log_dir = './log'

# Define the datasets and epsilon values you want to process
datasets = ["femnist", "fashion-mnist", "shakespeare"]
epsilons = ["0.01", "0.1", "1.0", "10.0", "25.0", "50.0", "none"]

# Function to convert log file to CSV
def convert_log_to_csv(log_filename, csv_filename):
    with open(log_filename, 'r') as log_file, open(csv_filename, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Round', 'Accuracy'])  # Write the header

        for line in log_file:
            if "Round" in line and "Global Model Accuracy" in line:
                parts = line.split()
                round_num = parts[1].split('/')[0]  # Extract the round number
                accuracy = parts[-1].strip('%')  # Extract the accuracy
                csv_writer.writerow([round_num, accuracy])

# Iterate over each dataset and epsilon value
for dataset in datasets:
    for epsilon in epsilons:
        # Construct the log file name
        epsilon_str = epsilon if epsilon != "none" else "None"
        log_filename = os.path.join(log_dir, f'{dataset}_{epsilon}_training.log')
        csv_filename = os.path.join(log_dir, f'{dataset}_{epsilon}_accuracy_data.csv')

        # Check if the log file exists
        if os.path.exists(log_filename):
            print(f"Converting {log_filename} to {csv_filename}")
            convert_log_to_csv(log_filename, csv_filename)
        else:
            print(f"Log file {log_filename} does not exist. Skipping...")

print("Conversion complete.")