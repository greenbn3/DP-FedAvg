import os
import csv
import re

# Define the directory containing the log files
log_dir = './log'

# Define the datasets and epsilon values you want to process
datasets = ["femnist", "fashion-mnist", "shakespeare"]
epsilons = ["0.01", "0.1", "0.5", "1.0", "10.0", "25.0", "50.0", "none"]

# Regular expressions to match Round and Accuracy lines
round_regex = re.compile(r"Round\s+(\d+)/\d+")
accuracy_regex = re.compile(r"Global Model Accuracy:\s+(\d+\.\d+)%")

# Function to convert log file to CSV
def convert_log_to_csv(log_filename, csv_filename):
    with open(log_filename, 'r') as log_file, open(csv_filename, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Round', 'Accuracy'])  # Write the header

        current_round = None

        for line in log_file:
            line = line.strip()

            # Check for Round line
            round_match = round_regex.match(line)
            if round_match:
                current_round = int(round_match.group(1))
                continue  # Move to next line to find accuracy

            # Check for Accuracy line
            accuracy_match = accuracy_regex.match(line)
            if accuracy_match and current_round is not None:
                accuracy = float(accuracy_match.group(1))
                csv_writer.writerow([current_round, accuracy])
                current_round = None  # Reset for the next round
                continue

            # Ignore other lines

    print(f"Finished processing {log_filename} and saved to {csv_filename}")

# Iterate over each dataset and epsilon value
for dataset in datasets:
    for epsilon in epsilons:
        # Handle 'none' epsilon consistently in filenames
        epsilon_str = 'none' if epsilon.lower() == "none" else epsilon

        # Construct the log file name
        log_filename = os.path.join(log_dir, f'{dataset}_{epsilon_str}_training.log')
        csv_filename = os.path.join(log_dir, f'{dataset}_{epsilon_str}_accuracy_data.csv')

        # Check if the log file exists
        if os.path.exists(log_filename):
            print(f"Converting '{log_filename}' to '{csv_filename}'")
            convert_log_to_csv(log_filename, csv_filename)
        else:
            print(f"Log file '{log_filename}' does not exist. Skipping...")

print("All conversions completed.")