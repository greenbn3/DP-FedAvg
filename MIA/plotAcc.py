import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

def plot_training_progress(dataset, log_dir="./gaussian_log", output_filename=None):
    """
    Reads CSV files in the log_dir with names of the form:
      {dataset}_training_progress_epsilon_{epsilon}.csv
    and plots the training accuracy (per round) for all epsilon values on a single plot.
    
    Parameters:
      dataset (str): Name of the dataset (e.g., "mnist" or "cifar10").
      log_dir (str): Directory where the CSV log files are stored.
      output_filename (str): (Optional) Filename for saving the plot.
    """
    # Create a glob pattern to locate all CSV files for this dataset
    pattern = os.path.join(log_dir, f"{dataset}_training_progress_epsilon_*.csv")
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        print(f"No CSV files found matching pattern {pattern}")
        return

    plt.figure(figsize=(10, 6))
    
    # Loop through each CSV file
    for csv_file in csv_files:
        # Extract epsilon value from the filename.
        # Expected filename format: {dataset}_training_progress_epsilon_{epsilon}.csv
        base_name = os.path.basename(csv_file)
        # Split by "_" and take the last element (before the .csv)
        epsilon_str = base_name.split("_")[-1].replace(".csv", "")
        
        # Read CSV into DataFrame (assumes columns: "Round", "Accuracy")
        df = pd.read_csv(csv_file)
        
        plt.plot(df["Round"], df["Accuracy"], marker="o", label=f"ε = {epsilon_str}")

    plt.xlabel("Training Rounds")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{dataset.upper()} Training Progress for Different ε Values")
    plt.legend()
    plt.grid(True)
    
    if output_filename is None:
        output_filename = os.path.join(log_dir, f"{dataset}_training_progress_all_epsilons.png")
    
    plt.savefig(output_filename)
    print(f"Plot saved as {output_filename}")
    plt.show()

if __name__ == "__main__":
    # Set your dataset (e.g., "mnist" or "cifar10")
    dataset = "cifar10"
    plot_training_progress(dataset)
