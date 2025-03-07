import os
import glob
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import platform

# --- Model Definitions ---

def mnist_model():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )

def fashionmnist_model():
    return mnist_model()

def cifar10_model():
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(4096, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )

# --- Dataset Helper Functions ---

def get_mnist_datasets():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return train_ds, test_ds

def get_fashionmnist_datasets():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_ds = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_ds = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    return train_ds, test_ds

def get_cifar10_datasets():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_ds = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_ds = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    return train_ds, test_ds

# --- Threshold-Based Membership Inference Attack Implementation ---
def threshold_attack(model, train_dataset, test_dataset, device, num_samples=1000):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    
    # Subsample the training and test sets.
    train_subset, _ = random_split(train_dataset, [num_samples, len(train_dataset) - num_samples])
    test_subset, _ = random_split(test_dataset, [num_samples, len(test_dataset) - num_samples])
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)
    
    train_losses = []
    test_losses = []
    
    with torch.no_grad():
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            logits = model(data)
            losses = loss_fn(logits, labels)
            train_losses.extend(losses.cpu().numpy())
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            logits = model(data)
            losses = loss_fn(logits, labels)
            test_losses.extend(losses.cpu().numpy())
    
    threshold = np.mean(train_losses)
    
    train_preds = (np.array(train_losses) < threshold).astype(int)
    test_preds = (np.array(test_losses) < threshold).astype(int)
    
    member_labels = np.ones(len(train_preds), dtype=int)
    nonmember_labels = np.zeros(len(test_preds), dtype=int)
    
    correct = np.sum(train_preds == member_labels) + np.sum(test_preds == nonmember_labels)
    total = len(train_preds) + len(test_preds)
    attack_accuracy = correct / total * 100
    return attack_accuracy, threshold, np.mean(train_losses), np.mean(test_losses)

# --- Device Selection ---
if platform.system() in ["Windows", "Linux"]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
elif platform.system() == "Darwin":
    device = torch.device("mps" if torch.has_mps else "cpu")
else:
    device = torch.device("cpu")
print("Using device:", device)

# --- Main: Process Saved Models and Save Attack Results to CSV ---
log_folder = "./log"  # Folder where .pt files are stored
pt_files = glob.glob(os.path.join(log_folder, "*_global_model_epsilon_*.pt"))
output_csv = os.path.join(log_folder, "mia_threshold_attack_results.csv")

results = []

for pt_file in pt_files:
    basename = os.path.basename(pt_file)  # e.g., "mnist_global_model_epsilon_none.pt"
    parts = basename.split('_')
    dataset_name = parts[0].lower()
    epsilon_str = parts[-1].replace(".pt", "")
    
    print(f"Processing file: {pt_file} for dataset: {dataset_name}, epsilon: {epsilon_str}")
    
    # Select the appropriate model function and dataset
    if dataset_name == "mnist":
        model_fn = mnist_model
        train_ds, test_ds = get_mnist_datasets()
    elif dataset_name == "fashionmnist":
        model_fn = fashionmnist_model
        train_ds, test_ds = get_fashionmnist_datasets()
    elif dataset_name == "cifar10":
        model_fn = cifar10_model
        train_ds, test_ds = get_cifar10_datasets()
    else:
        print(f"Unsupported dataset: {dataset_name}. Skipping file: {pt_file}")
        continue
    
    # Load the model.
    model = model_fn().to(device)
    state_dict = torch.load(pt_file, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Run the threshold-based membership inference attack.
    attack_acc, threshold, avg_train_loss, avg_test_loss = threshold_attack(model, train_ds, test_ds, device)
    
    results.append([dataset_name, epsilon_str, attack_acc, threshold, avg_train_loss, avg_test_loss])
    print(f"Dataset: {dataset_name}, Epsilon: {epsilon_str}, Attack Accuracy: {attack_acc:.2f}%")
    
# --- Sorting the Results ---
# Sort by dataset and epsilon in decreasing order.
# For epsilon, if it's "none", treat it as infinity.
def epsilon_to_float(eps_str):
    try:
        return float(eps_str)
    except ValueError:
        return float('inf')

results = sorted(results, key=lambda row: (row[0], epsilon_to_float(row[1])), reverse=True)

# Save all results to CSV.
with open(output_csv, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Dataset", "Epsilon", "Threshold Attack Accuracy", "Loss Threshold", "Avg Train Loss", "Avg Test Loss"])
    writer.writerows(results)

print(f"All attack results saved to: {output_csv}")