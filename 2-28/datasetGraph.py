import pandas as pd
import matplotlib.pyplot as plt

# Load the summary CSV
df = pd.read_csv('./log/mia_accuracy_summary_recreated.csv')

# Plot MIA Accuracy for each dataset
for dataset in df['Dataset'].unique():
    subset = df[df['Dataset'] == dataset]
    plt.figure(figsize=(10, 6))
    plt.plot(subset['Epsilon'], subset['MIA Accuracy'], marker='o')
    plt.xlabel('Epsilon')
    plt.ylabel('MIA Accuracy (%)')
    plt.title(f'MIA Accuracy vs Epsilon for {dataset}')
    plt.grid(True)
    plot_filename = f'./log/{dataset}_MIA_Accuracy_vs_Epsilon.png'
    plt.savefig(plot_filename)
    plt.close()
    print(f"MIA Accuracy plot saved as {plot_filename}.")

print("All MIA Accuracy plots generated successfully.")