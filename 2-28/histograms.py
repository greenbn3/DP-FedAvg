import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import platform

# --- Updated Model Definitions ---

def mnist_model():
    """Simple feedforward network for MNIST (and FashionMNIST)"""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )

def fashionmnist_model():
    """Same architecture as MNIST for FashionMNIST"""
    return mnist_model()

def cifar10_model():
    """
    Updated CIFAR-10 model architecture matching the saved state.
    Architecture:
      - Conv2d(3, 32, kernel_size=3, padding=1) -> ReLU -> MaxPool2d(kernel_size=2)
      - Conv2d(32, 64, kernel_size=3, padding=1) -> ReLU -> MaxPool2d(kernel_size=2)
      - Flatten (64*8*8 = 4096)
      - Linear(4096, 128) -> ReLU
      - Linear(128, 64) -> ReLU
      - Linear(64, 10)
    """
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(4096, 128),  # Updated: output 128 instead of 256
        nn.ReLU(),
        nn.Linear(128, 64),    # New intermediate layer
        nn.ReLU(),
        nn.Linear(64, 10)      # Final layer: 10 classes
    )

# --- Dataset Helper Functions ---

def get_mnist_datasets():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    return train_dataset, test_dataset

def get_fashionmnist_datasets():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
    return train_dataset, test_dataset

def get_cifar10_datasets():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    return train_dataset, test_dataset

# --- Function to Generate and Save Histograms ---

def generate_histograms(model, dataset, device, output_prefix):
    model.eval()
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    all_max_softmax = []
    all_max_logits = []
    
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            logits = model(data)
            softmax = F.softmax(logits, dim=1)
            max_softmax, _ = softmax.max(dim=1)
            max_logits, _ = logits.max(dim=1)
            all_max_softmax.extend(max_softmax.cpu().numpy())
            all_max_logits.extend(max_logits.cpu().numpy())
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(all_max_softmax, bins=50, color='blue', alpha=0.7)
    plt.title("Histogram of Max Softmax Probabilities")
    plt.xlabel("Max Softmax Value")
    plt.ylabel("Frequency")
    
    plt.subplot(1, 2, 2)
    plt.hist(all_max_logits, bins=50, color='green', alpha=0.7)
    plt.title("Histogram of Max Logit Values")
    plt.xlabel("Max Logit Value")
    plt.ylabel("Frequency")
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}.png")
    plt.close()

# --- Device Selection ---

if platform.system() in ["Windows", "Linux"]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
elif platform.system() == "Darwin":
    device = torch.device("mps" if torch.has_mps else "cpu")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# --- Main Loop: Process Saved .pt Files and Generate Histograms ---

log_folder = "./log"  # Folder where the .pt files are stored
pt_files = glob.glob(os.path.join(log_folder, "*_global_model_epsilon_*.pt"))

for pt_file in pt_files:
    basename = os.path.basename(pt_file)  # e.g. "mnist_global_model_epsilon_none.pt"
    parts = basename.split('_')
    
    # Extract dataset name (first part) and epsilon value (last part without .pt)
    dataset_name = parts[0].lower()
    epsilon_str = parts[-1].replace(".pt", "")
    
    print(f"Processing file: {pt_file} for dataset: {dataset_name}, epsilon: {epsilon_str}")
    
    # Choose the correct model function and dataset based on the dataset name
    if dataset_name == "mnist":
        model_fn = mnist_model
        train_dataset, test_dataset = get_mnist_datasets()
    elif dataset_name == "fashionmnist":
        model_fn = fashionmnist_model
        train_dataset, test_dataset = get_fashionmnist_datasets()
    elif dataset_name == "cifar10":
        model_fn = cifar10_model
        train_dataset, test_dataset = get_cifar10_datasets()
    else:
        print(f"Unsupported dataset: {dataset_name}. Skipping file: {pt_file}")
        continue
    
    # Load the model from the .pt file
    model = model_fn().to(device)
    state_dict = torch.load(pt_file, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Define output prefixes for histogram images
    output_prefix_train = os.path.join(log_folder, f"{dataset_name}_epsilon_{epsilon_str}_members_histogram")
    output_prefix_test = os.path.join(log_folder, f"{dataset_name}_epsilon_{epsilon_str}_nonmembers_histogram")
    
    # Generate and save histograms for training (members) and test (non-members) datasets
    generate_histograms(model, train_dataset, device, output_prefix_train)
    generate_histograms(model, test_dataset, device, output_prefix_test)
    
    print(f"Histograms saved for {pt_file} as:\n  {output_prefix_train}.png\n  {output_prefix_test}.png")

print("All histograms generated and saved.")