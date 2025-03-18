import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

def plot_mia_metrics_for_file(csv_path, output_dir="./gaussian_log"):
    """
    Reads a CSV file with name format {dataset}_mia_summary.csv and
    plots MIA metrics versus epsilon.
    
    Expected columns in the CSV:
      - Epsilon
      - FinalGlobalAccuracy
      - MIA_InSet_Accuracy
      - MIA_OutSet_Accuracy
      - MIA_Overall_Accuracy
    """
    # Extract dataset name from filename
    base_name = os.path.basename(csv_path)
    dataset_name = base_name.split("_")[0]  # e.g., "mnist" from "mnist_mia_summary.csv"
    
    # Read CSV into DataFrame
    df = pd.read_csv(csv_path)
    
    # Ensure epsilon values are sorted. For non-numeric entries, assign a value like -1.
    def convert_epsilon(eps):
        try:
            return float(eps)
        except:
            return -1.0
    df["epsilon_numeric"] = df["Epsilon"].apply(convert_epsilon)
    df.sort_values("epsilon_numeric", inplace=True)
    
    eps_labels = df["Epsilon"].tolist()
    
    plt.figure(figsize=(10, 6))
    plt.plot(eps_labels, df["FinalGlobalAccuracy"], marker='o', label="Final Global Acc.")
    plt.plot(eps_labels, df["MIA_InSet_Accuracy"], marker='o', label="MIA In-Set Acc.")
    plt.plot(eps_labels, df["MIA_OutSet_Accuracy"], marker='o', label="MIA Out-Set Acc.")
    plt.plot(eps_labels, df["MIA_Overall_Accuracy"], marker='o', label="MIA Overall Acc.")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy (%)")
    plt.title(f"MIA Metrics vs Epsilon for {dataset_name.upper()}")
    plt.legend()
    plt.grid(True)
    
    output_filename = os.path.join(output_dir, f"{dataset_name}_MIA_Metrics_vs_Epsilon.png")
    plt.savefig(output_filename)
    print(f"MIA metrics plot saved as {output_filename}")
    plt.close()

def main():
    log_dir = "./gaussian_log"  # Change if necessary
    # Find all CSV files that follow the pattern {dataset}_mia_summary.csv
    csv_files = glob.glob(os.path.join(log_dir, "*_mia_summary.csv"))
    
    if not csv_files:
        print("No MIA summary CSV files found in the log directory.")
        return

    for csv_file in csv_files:
        plot_mia_metrics_for_file(csv_file, output_dir=log_dir)

if __name__ == "__main__":
    main()