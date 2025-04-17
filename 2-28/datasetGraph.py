# import pandas as pd
# import matplotlib.pyplot as plt

# # Load the summary CSV
# df = pd.read_csv('./log/mia_accuracy_summary_recreated.csv')

# # Plot MIA Accuracy for each dataset
# for dataset in df['Dataset'].unique():
#     subset = df[df['Dataset'] == dataset]
#     plt.figure(figsize=(10, 6))
#     plt.plot(subset['Epsilon'], subset['Loss Attack Overall Accuracy'], marker='o')
#     plt.xlabel('Epsilon')
#     plt.ylabel('MIA Accuracy (%)')
#     plt.title(f'(Loss) MIA Accuracy vs Epsilon for {dataset}')
#     plt.grid(True)
#     plot_filename = f'./log/{dataset}_Loss_MIA_Accuracy_vs_Epsilon.png'
#     plt.savefig(plot_filename)
#     plt.close()
#     print(f"MIA Accuracy plot saved as {plot_filename}.")

# print("All MIA Accuracy plots generated successfully.")


import pandas as pd
import matplotlib.pyplot as plt

# Load the summary CSV
df = pd.read_csv('./log/mia_accuracy_summary_recreated.csv')

# Combined plot for Loss Attack
plt.figure(figsize=(10, 6))
for dataset in df['Dataset'].unique():
    subset = df[df['Dataset'] == dataset]
    plt.plot(subset['Epsilon'], subset['Loss Attack Overall Accuracy'],
             marker='o', label=dataset)
plt.xlabel('Epsilon')
plt.ylabel('MIA Accuracy (%)')
plt.title('Combined Loss Attack: MIA Accuracy vs Epsilon')
plt.legend(title='Dataset')
plt.grid(True)
plt.savefig('./log/Combined_Loss_MIA_Accuracy_vs_Epsilon.png')
plt.close()

# Combined plot for ART Attack
plt.figure(figsize=(10, 6))
for dataset in df['Dataset'].unique():
    subset = df[df['Dataset'] == dataset]
    plt.plot(subset['Epsilon'], subset['ART MIA Accuracy'],
             marker='o', label=dataset)
plt.xlabel('Epsilon')
plt.ylabel('MIA Accuracy (%)')
plt.title('Combined ART Attack: MIA Accuracy vs Epsilon')
plt.legend(title='Dataset')
plt.grid(True)
plt.savefig('./log/Combined_ART_MIA_Accuracy_vs_Epsilon.png')
plt.close()

print("Combined plots saved as 'Combined_Loss_MIA_Accuracy_vs_Epsilon.png' and 'Combined_ART_MIA_Accuracy_vs_Epsilon.png'.")