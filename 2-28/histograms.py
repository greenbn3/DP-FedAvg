import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
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
        nn.Linear(64, 10),
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
    train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    return train, test

def get_fashionmnist_datasets():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    test  = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
    return train, test

def get_cifar10_datasets():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test  = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    return train, test

# --- Helper Functions ---

def find_global_max_freq(model, dataset, device):
    model.eval()
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    all_softmax = []
    all_logits = []
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            logits = model(data)
            softmax = F.softmax(logits, dim=1)
            max_s, _ = softmax.max(dim=1)
            max_l, _ = logits.max(dim=1)
            all_softmax.extend(max_s.cpu().numpy())
            all_logits.extend(max_l.cpu().numpy())
    softmax_hist, _ = np.histogram(all_softmax, bins=50)
    logits_hist,  _ = np.histogram(all_logits,  bins=50)
    return softmax_hist.max(), logits_hist.max()

def generate_histograms(model, dataset, device, output_prefix, dataset_name, epsilon_str, global_max_freq):
    model.eval()
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    all_softmax = []
    all_logits = []
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            logits = model(data)
            softmax = F.softmax(logits, dim=1)
            max_s, _ = softmax.max(dim=1)
            max_l, _ = logits.max(dim=1)
            all_softmax.extend(max_s.cpu().numpy())
            all_logits.extend(max_l.cpu().numpy())
    
    plt.figure(figsize=(12, 5))
    # Softmax histogram
    plt.subplot(1, 2, 1)
    plt.hist(all_softmax, bins=50, alpha=0.7)
    plt.title(f"{dataset_name.upper()} (ε={epsilon_str}) - Max Softmax")
    plt.xlabel("Max Softmax Value")
    plt.ylabel("Frequency")
    plt.ylim(0, global_max_freq["softmax"])
    
    # Logits histogram
    plt.subplot(1, 2, 2)
    plt.hist(all_logits, bins=50, alpha=0.7)
    plt.title(f"{dataset_name.upper()} (ε={epsilon_str}) - Max Logit")
    plt.xlabel("Max Logit Value")
    plt.ylabel("Frequency")
    plt.ylim(0, global_max_freq["logits"])
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
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

# --- Main Execution ---

log_folder = "./log"
log_histogram_folder = "./log_histogram"
os.makedirs(log_histogram_folder, exist_ok=True)
pt_files = glob.glob(os.path.join(log_folder, "*_global_model_epsilon_*.pt"))

# First pass: compute global max frequencies
global_softmax_max = 0
global_logits_max = 0
for pt_file in pt_files:
    basename = os.path.basename(pt_file)
    parts = basename.split('_')
    dataset_name = parts[0].lower()
    epsilon_str = parts[-1].replace(".pt", "")
    
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
        continue
    
    model = model_fn().to(device)
    model.load_state_dict(torch.load(pt_file, map_location=device))
    
    s_max, l_max = find_global_max_freq(model, train_ds, device)
    global_softmax_max = max(global_softmax_max, s_max)
    global_logits_max  = max(global_logits_max,  l_max)
    
    s_max, l_max = find_global_max_freq(model, test_ds, device)
    global_softmax_max = max(global_softmax_max, s_max)
    global_logits_max  = max(global_logits_max,  l_max)

global_max_freq = {"softmax": global_softmax_max, "logits": global_logits_max}

# Second pass: generate and save standardized histograms
for pt_file in pt_files:
    basename = os.path.basename(pt_file)
    parts = basename.split('_')
    dataset_name = parts[0].lower()
    epsilon_str = parts[-1].replace(".pt", "")
    
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
        continue
    
    model = model_fn().to(device)
    model.load_state_dict(torch.load(pt_file, map_location=device))
    
    prefix_train = os.path.join(log_histogram_folder,
        f"{dataset_name}_epsilon_{epsilon_str}_members_histogram")
    prefix_test  = os.path.join(log_histogram_folder,
        f"{dataset_name}_epsilon_{epsilon_str}_nonmembers_histogram")
    
    generate_histograms(model, train_ds, device, prefix_train,
                        dataset_name, epsilon_str, global_max_freq)
    generate_histograms(model, test_ds, device, prefix_test,
                        dataset_name, epsilon_str, global_max_freq)

print("All standardized histograms generated and saved.")