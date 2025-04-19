# Script to rerun loss-based MIA on MNIST, FashionMNIST, and CIFAR-10 saved models,
# generate confusion matrices (2x2), save CSVs of results, and create loss distribution plots

import torch
import torch.nn as nn
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Set matplotlib backend
import matplotlib
matplotlib.use("agg")

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model architectures ---
def mnist_model():
    return nn.Sequential(
        nn.Flatten(), nn.Linear(28*28, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10)
    )

def fashionmnist_model():
    return mnist_model()

def cifar10_model():
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
        nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(64 * 8 * 8, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 10)
    )

# --- Datasets ---
def get_datasets(name):
    if name == "mnist":
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset = datasets.MNIST("./data", train=True, download=True, transform=tf)
    elif name == "fashionmnist":
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    elif name == "cifar10":
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = datasets.CIFAR10("./data", train=True, download=True, transform=tf)
    else:
        raise ValueError("Unsupported dataset")

    member_set = Subset(dataset, list(range(0, 1000)))
    non_member_set = Subset(dataset, list(range(1000, 2000)))
    return DataLoader(member_set, batch_size=32, shuffle=False), DataLoader(non_member_set, batch_size=32, shuffle=False)

# --- Run Loss-Based MIA ---
def run_loss_based_attack(model, member_loader, non_member_loader, dataset_name, epsilon):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    member_losses, non_member_losses = [], []
    all_true, all_pred = [], []

    for data, target in member_loader:
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        losses = loss_fn(outputs, target)
        member_losses.extend(losses.detach().cpu().numpy())

    for data, target in non_member_loader:
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        losses = loss_fn(outputs, target)
        non_member_losses.extend(losses.detach().cpu().numpy())

    member_losses = np.array(member_losses)
    non_member_losses = np.array(non_member_losses)
    threshold = (member_losses.mean() + non_member_losses.mean()) / 2.0

    member_preds = (member_losses < threshold).astype(int)
    non_member_preds = (non_member_losses < threshold).astype(int)

    all_pred = np.concatenate([member_preds, non_member_preds])
    all_true = np.concatenate([np.ones_like(member_preds), np.zeros_like(non_member_preds)])

    cm = confusion_matrix(all_true, all_pred)
    acc = (all_pred == all_true).mean() * 100

    return cm, all_true, all_pred, member_losses, non_member_losses, threshold, acc

# --- Plotting ---
def save_plots_and_csv(dataset_name, epsilon, cm, true, pred, member_losses, non_member_losses, threshold):
    label = f"{dataset_name}_eps_{epsilon}"
    save_dir = f"./log/mia_results/{dataset_name}"
    os.makedirs(save_dir, exist_ok=True)

    # Save CSV of confusion matrix
    with open(os.path.join(save_dir, f"conf_matrix_{label}.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["", "Predicted 0", "Predicted 1"])
        writer.writerow(["Actual 0", cm[0][0], cm[0][1]])
        writer.writerow(["Actual 1", cm[1][0], cm[1][1]])

    # Save CSV of true/pred labels
    with open(os.path.join(save_dir, f"true_vs_pred_{label}.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["True Label", "Predicted Label"])
        writer.writerows(zip(true, pred))

    # Save confusion matrix plot
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-member", "Member"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"MIA Confusion Matrix ({label})")
    plt.savefig(os.path.join(save_dir, f"conf_matrix_plot_{label}.png"))
    plt.close()

    # Save loss distribution plot
    plt.hist(member_losses, bins=50, alpha=0.6, label='Member Loss')
    plt.hist(non_member_losses, bins=50, alpha=0.6, label='Non-member Loss')
    plt.axvline(threshold, color='k', linestyle='dashed', linewidth=1, label='Threshold')
    plt.legend()
    plt.title(f"Loss Distribution (Threshold={threshold:.4f})\n{label}")
    plt.xlabel("Cross Entropy Loss")
    plt.ylabel("Count")
    plt.savefig(os.path.join(save_dir, f"loss_distribution_{label}.png"))
    plt.close()

# --- Main Loop ---
def main():
    datasets_list = ["mnist", "fashionmnist", "cifar10"]
    epsilons = [None, 75.0, 50.0, 25.0, 10.0, 5.0, 1.0, 0.1, 0.01, 0.001]

    for dataset in datasets_list:
        member_loader, non_member_loader = get_datasets(dataset)
        model_fn = mnist_model if dataset == "mnist" else fashionmnist_model if dataset == "fashionmnist" else cifar10_model

        for eps in epsilons:
            eps_str = "none" if eps is None else str(eps)
            pt_file = f"./log/{dataset}_global_model_epsilon_{eps_str}.pt"
            if not os.path.exists(pt_file):
                print(f"Skipping missing model: {pt_file}")
                continue

            model = model_fn().to(device)
            model.load_state_dict(torch.load(pt_file, map_location=device))
            cm, true, pred, mem_loss, nonmem_loss, threshold, acc = run_loss_based_attack(
                model, member_loader, non_member_loader, dataset, eps_str)

            print(f"[{dataset.upper()} - ε={eps_str}] MIA Accuracy: {acc:.2f}%")
            save_plots_and_csv(dataset, eps_str, cm, true, pred, mem_loss, nonmem_loss, threshold)

if __name__ == "__main__":
    main()
