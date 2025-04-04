import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np

# --- MODEL DEFINITIONS ---
def mnist_model():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )

def cifar10_model():
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(64 * 8 * 8, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )

def fashionmnist_model():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )

# --- DATASET LOADING FUNCTIONS ---
def get_mnist_datasets():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    full_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    return train_dataset, test_dataset

def get_cifar10_datasets():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    full_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    return train_dataset, test_dataset

def get_fashionmnist_datasets():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    full_dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    return train_dataset, test_dataset

# --- GLOBAL MODEL EVALUATION ---
def evaluate_global_model(model, test_dataset, device):
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)
    accuracy = 100 * correct / total
    return accuracy

# --- ART-BASED MEMBERSHIP INFERENCE ATTACK ---
class MembershipInferenceAttack:
    def __init__(self, shadow_model, target_model, device):
        self.shadow_model = shadow_model
        self.target_model = target_model
        self.attack_model = nn.Sequential(
            nn.Linear(10, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        ).to(device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.attack_model.parameters(), lr=0.001)
        self.device = device

    def generate_attack_data(self, data_loader, model, label):
        attack_features = []
        attack_labels = []
        model.eval()
        with torch.no_grad():
            for data, _ in data_loader:
                data = data.to(self.device)
                outputs = model(data)
                attack_features.append(outputs)
                attack_labels.extend([label] * outputs.shape[0])
        attack_features = torch.cat(attack_features)
        attack_labels = torch.tensor(attack_labels).to(self.device)
        return attack_features, attack_labels

    def train_attack_model(self, member_loader, non_member_loader):
        in_feats, in_labels = self.generate_attack_data(member_loader, self.shadow_model, label=1)
        out_feats, out_labels = self.generate_attack_data(non_member_loader, self.shadow_model, label=0)
        attack_features = torch.cat((in_feats, out_feats))
        attack_labels = torch.cat((in_labels, out_labels))
        
        permutation = torch.randperm(attack_features.size(0))
        attack_features = attack_features[permutation]
        attack_labels = attack_labels[permutation]
        
        self.attack_model.train()
        batch_size = 32
        num_epochs = 5
        for epoch in range(num_epochs):
            for i in range(0, attack_features.size(0), batch_size):
                batch_features = attack_features[i:i+batch_size]
                batch_labels = attack_labels[i:i+batch_size]
                self.optimizer.zero_grad()
                preds = self.attack_model(batch_features)
                loss = self.loss_fn(preds, batch_labels)
                loss.backward()
                self.optimizer.step()

    def evaluate_attack(self, member_loader, non_member_loader):
        self.attack_model.eval()
        true_labels = []
        predicted_labels = []
        
        with torch.no_grad():
            for data, _ in member_loader:
                data = data.to(self.device)
                outputs = self.target_model(data)
                preds = self.attack_model(outputs)
                _, predicted = torch.max(preds, 1)
                predicted_labels.extend(predicted.cpu().numpy())
                true_labels.extend([1] * data.size(0))
            for data, _ in non_member_loader:
                data = data.to(self.device)
                outputs = self.target_model(data)
                preds = self.attack_model(outputs)
                _, predicted = torch.max(preds, 1)
                predicted_labels.extend(predicted.cpu().numpy())
                true_labels.extend([0] * data.size(0))
        
        tp = sum((p == 1 and t == 1) for p, t in zip(predicted_labels, true_labels))
        fp = sum((p == 1 and t == 0) for p, t in zip(predicted_labels, true_labels))
        fn = sum((p == 0 and t == 1) for p, t in zip(predicted_labels, true_labels))
        tn = sum((p == 0 and t == 0) for p, t in zip(predicted_labels, true_labels))
        
        accuracy = (tp + tn) / len(true_labels) * 100
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        return accuracy, tpr, fpr, precision

# --- LOSS-BASED MEMBERSHIP INFERENCE ATTACK ---
def run_loss_based_attack(target_model, member_loader, non_member_loader, device):
    target_model.eval()
    attack_loss_fn = nn.CrossEntropyLoss(reduction='none')
    
    member_losses = []
    for data, target in member_loader:
        data, target = data.to(device), target.to(device)
        outputs = target_model(data)
        losses = attack_loss_fn(outputs, target)
        member_losses.extend(losses.detach().cpu().numpy())
    
    non_member_losses = []
    for data, target in non_member_loader:
        data, target = data.to(device), target.to(device)
        outputs = target_model(data)
        losses = attack_loss_fn(outputs, target)
        non_member_losses.extend(losses.detach().cpu().numpy())
    
    member_losses = np.array(member_losses)
    non_member_losses = np.array(non_member_losses)
    
    avg_member_loss = member_losses.mean()
    avg_non_member_loss = non_member_losses.mean()
    threshold = (avg_member_loss + avg_non_member_loss) / 2.0
    member_preds = (member_losses < threshold).astype(int)
    non_member_preds = (non_member_losses < threshold).astype(int)
    overall_accuracy = ((member_preds == 1).sum() + (non_member_preds == 0).sum()) / (len(member_losses) + len(non_member_losses)) * 100 
    return overall_accuracy, avg_member_loss, avg_non_member_loss, threshold

# --- SORTING HELPER ---
def get_sort_key(filename):
    """
    Expects filename of the format: {dataset}_global_model_epsilon_{epsilon}.pt
    Returns a tuple (dataset, -order) where order is defined by the custom ranking.
    """
    # Split the filename to get dataset and epsilon string
    try:
        parts = filename.split("_global_model_epsilon_")
        dataset = parts[0]
        epsilon_str = parts[1].replace(".pt", "")
    except IndexError:
        dataset, epsilon_str = filename, ""
    
    # Define the desired order: higher number means higher rank.
    epsilon_order = {
        "none": 10,
        "75.0": 9,
        "50.0": 8,
        "25.0": 7,
        "10.0": 6,
        "5.0": 5,
        "1.0": 4,
        "0.1": 3,
        "0.01": 2,
        "0.001": 1
    }
    order = epsilon_order.get(epsilon_str, 0)
    # We use negative order so that sorting in ascending order gives descending epsilon order.
    return (dataset, -order)

# --- MAIN SCRIPT ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = "../2-28/log"  # Directory containing the .pt files
    output_csv = os.path.join(log_dir, "mia_accuracy_summary_recreated.csv")
    
    # Find files matching the pattern "*_global_model_epsilon_*.pt"
    pt_files = [f for f in os.listdir(log_dir) if f.endswith(".pt") and "global_model" in f]
    
    # Sort the files by dataset and epsilon order
    pt_files.sort(key=get_sort_key)
    
    with open(output_csv, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "Dataset", "Epsilon", 
            "ART MIA Accuracy", "TPR", "FPR", "Precision",
            "Loss Attack Overall Accuracy", "Avg Member Loss", "Avg Non-Member Loss", "Loss Attack Threshold",
            "Global Model Accuracy"
        ])
        
        for pt_file in pt_files:
            # Expected filename format: "{dataset}_global_model_epsilon_{epsilon}.pt"
            parts = pt_file.split("_global_model_epsilon_")
            dataset_choice = parts[0]
            epsilon_str = parts[1].replace(".pt", "")
            print(f"Processing file: {pt_file} | Dataset: {dataset_choice} | Epsilon: {epsilon_str}")
            
            # Select the appropriate model function and datasets
            if dataset_choice == "mnist":
                model_fn = mnist_model
                train_dataset, test_dataset = get_mnist_datasets()
            elif dataset_choice == "cifar10":
                model_fn = cifar10_model
                train_dataset, test_dataset = get_cifar10_datasets()
            elif dataset_choice == "fashionmnist":
                model_fn = fashionmnist_model
                train_dataset, test_dataset = get_fashionmnist_datasets()
            else:
                print(f"Unsupported dataset: {dataset_choice}")
                continue
            
            # Load the saved global model state_dict
            model = model_fn().to(device)
            model_path = os.path.join(log_dir, pt_file)
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            
            # Evaluate global model accuracy on the test dataset
            global_accuracy = evaluate_global_model(model, test_dataset, device)
            
            # Prepare member and non-member DataLoaders (using 1,000 samples each)
            member_size = 1000
            non_member_size = 1000
            member_indices = list(range(member_size))
            non_member_indices = list(range(member_size, member_size + non_member_size))
            
            member_subset = Subset(train_dataset, member_indices)
            non_member_subset = Subset(test_dataset, non_member_indices)
            member_loader = DataLoader(member_subset, batch_size=32, shuffle=True)
            non_member_loader = DataLoader(non_member_subset, batch_size=32, shuffle=True)
            
            # Run ART-based membership inference attack
            shadow_model = model_fn().to(device)
            mia = MembershipInferenceAttack(shadow_model, model, device)
            mia.train_attack_model(member_loader, non_member_loader)
            art_accuracy, tpr, fpr, precision = mia.evaluate_attack(member_loader, non_member_loader)
            
            # Run loss-based membership inference attack
            loss_attack_accuracy, avg_member_loss, avg_non_member_loss, loss_threshold = run_loss_based_attack(
                model, member_loader, non_member_loader, device)
            
            # Write all the results to the CSV file
            writer.writerow([
                dataset_choice, epsilon_str,
                f"{art_accuracy:.2f}", f"{tpr:.3f}", f"{fpr:.3f}", f"{precision:.3f}",
                f"{loss_attack_accuracy:.2f}", f"{avg_member_loss:.4f}", f"{avg_non_member_loss:.4f}", f"{loss_threshold:.4f}",
                f"{global_accuracy:.2f}"
            ])
            
            print(f"Processed {pt_file}: Global Accuracy: {global_accuracy:.2f}%, ART MIA: {art_accuracy:.2f}%")
    
    print(f"\nSummary CSV recreated at {output_csv}")

if __name__ == "__main__":
    main()