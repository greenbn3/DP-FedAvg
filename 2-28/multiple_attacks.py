import os
import csv
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant
import platform
import numpy as np

# Set matplotlib backend explicitly for non-GUI use
import matplotlib
matplotlib.use("agg")

# Determine device based on OS and CUDA availability
if platform.system() in ["Windows", "Linux"]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
elif platform.system() == "Darwin":  # MacOS
    device = torch.device("mps" if torch.has_mps else "cpu")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")


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


# --- FEDERATED LEARNING CLIENT ---

class Client:
    def __init__(self, model, dataset, batch_size, learning_rate, device, epsilon=None, delta=1e-5):
        self.model = model().to(device)
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        self.device = device
        self.epsilon = epsilon
        self.delta = delta

        # Apply DP if epsilon is provided and not "none"
        if self.epsilon and self.epsilon != "none":
            self.privacy_engine = PrivacyEngine()
            self.model, self.optimizer, self.dataloader = self.privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.dataloader,
                noise_multiplier=self._calculate_noise_multiplier(),
                max_grad_norm=1.0,  # Adjust this if needed (e.g., to 0.1) for extreme DP
            )

    def _calculate_noise_multiplier(self):
        if self.epsilon and isinstance(self.epsilon, (float, int)):
            return 1.0 / self.epsilon
        return None

    def train(self, epochs):
        self.model.train()
        for epoch in range(epochs):
            for data, target in self.dataloader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_fn(output, target)
                loss.backward()
                self.optimizer.step()

    def get_weights(self):
        if hasattr(self.model, "_module"):
            return self.model._module.state_dict()
        else:
            return self.model.state_dict()

    def set_weights(self, state_dict):
        if hasattr(self.model, "_module"):
            self.model._module.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)


# --- FEDERATED LEARNING WITH DP ---

class FederatedLearningWithDP:
    def __init__(self, clients, model, dataset_choice, epsilon, delta):
        self.clients = clients
        self.global_model = model().to(device)
        self.dataset_choice = dataset_choice
        self.epsilon = epsilon
        self.delta = delta
        self.device = device  # needed for weight noise addition and evaluation
        self.privacy_accountant = RDPAccountant()
        self.noise_multiplier = 1.0 / epsilon if epsilon and epsilon != "none" else None

    def average_weights_with_noise(self, weights_list):
        avg_weights = {}
        for key in weights_list[0].keys():
            avg_weights[key] = torch.stack([weights[key] for weights in weights_list], dim=0).mean(dim=0)
        return avg_weights

    def train(self, rounds, epochs, test_dataset):
        global_accuracies = []
        for rnd in range(rounds):
            print(f"Round {rnd+1}/{rounds}")
            client_weights = []
            for client in self.clients:
                client.set_weights(self.global_model.state_dict())
                client.train(epochs)
                client_weights.append(client.get_weights())
            avg_weights = self.average_weights_with_noise(client_weights)
            self.global_model.load_state_dict(avg_weights)
            accuracy = self.evaluate_global_model(test_dataset)
            global_accuracies.append(accuracy)
        return global_accuracies

    def evaluate_global_model(self, test_dataset):
        self.global_model.eval()
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                _, predicted = torch.max(output, 1)
                correct += (predicted == target).sum().item()
                total += target.size(0)
        accuracy = 100 * correct / total
        print(f"Global Model Accuracy: {accuracy:.2f}%")
        return accuracy


# --- ART MEMBERSHIP INFERENCE ATTACK (existing) ---

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
                attack_labels.extend([label] * len(outputs))
        return torch.cat(attack_features), torch.tensor(attack_labels).to(self.device)

    def train_attack_model(self, member_loader, non_member_loader):
        in_feats, in_labels = self.generate_attack_data(member_loader, self.shadow_model, label=1)
        out_feats, out_labels = self.generate_attack_data(non_member_loader, self.shadow_model, label=0)
        attack_features = torch.cat((in_feats, out_feats))
        attack_labels = torch.cat((in_labels, out_labels))
        
        # Shuffle attack data
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
        
        # Process member data
        with torch.no_grad():
            for data, _ in member_loader:
                data = data.to(self.device)
                outputs = self.target_model(data)
                preds = self.attack_model(outputs)
                _, predicted = torch.max(preds, 1)
                predicted_labels.extend(predicted.cpu().numpy())
                true_labels.extend([1] * len(predicted))
        
        # Process non-member data
        with torch.no_grad():
            for data, _ in non_member_loader:
                data = data.to(self.device)
                outputs = self.target_model(data)
                preds = self.attack_model(outputs)
                _, predicted = torch.max(preds, 1)
                predicted_labels.extend(predicted.cpu().numpy())
                true_labels.extend([0] * len(predicted))
        
        # Calculate metrics
        tp = sum((p == 1 and t == 1) for p, t in zip(predicted_labels, true_labels))
        fp = sum((p == 1 and t == 0) for p, t in zip(predicted_labels, true_labels))
        fn = sum((p == 0 and t == 1) for p, t in zip(predicted_labels, true_labels))
        tn = sum((p == 0 and t == 0) for p, t in zip(predicted_labels, true_labels))
        
        accuracy = (tp + tn) / len(true_labels) * 100
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        return accuracy, tpr, fpr, precision


# --- LOSS-BASED MEMBERSHIP INFERENCE ATTACK (NEW) ---

def run_loss_based_attack(target_model, member_loader, non_member_loader, device):
    """
    Computes per-sample cross-entropy loss (with reduction='none') on the target model,
    then uses the average of member and non-member losses to define a threshold.
    Samples with loss below the threshold are classified as members.
    
    Returns:
      overall_accuracy: Overall accuracy of this threshold attack.
      avg_member_loss: Average loss on member data.
      avg_non_member_loss: Average loss on non-member data.
      threshold: The chosen threshold value.
    """
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
    
    # Define threshold as the midpoint between average member and non-member loss
    threshold = (avg_member_loss + avg_non_member_loss) / 2.0
    
    # Classify: if loss < threshold, predict member (1); else non-member (0)
    member_preds = (member_losses < threshold).astype(int)
    non_member_preds = (non_member_losses < threshold).astype(int)
    
    member_accuracy = (member_preds == 1).mean() * 100
    non_member_accuracy = (non_member_preds == 0).mean() * 100
    overall_accuracy = ((member_preds == 1).sum() + (non_member_preds == 0).sum()) / (len(member_losses) + len(non_member_losses)) * 100 
    
    
    return overall_accuracy, avg_member_loss, avg_non_member_loss, threshold


# --- DATASET HELPERS ---

def get_mnist_datasets():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    full_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    return train_dataset, test_dataset

def get_cifar10_datasets():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    full_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    return train_dataset, test_dataset

def get_fashionmnist_datasets():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    full_dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    return train_dataset, test_dataset

def distribute_data_among_clients(train_dataset, num_clients):
    client_datasets = []
    client_size = len(train_dataset) // num_clients
    indices = list(range(len(train_dataset)))
    random.shuffle(indices)
    for i in range(num_clients):
        client_indices = indices[i * client_size:(i + 1) * client_size]
        client_datasets.append(Subset(train_dataset, client_indices))
    return client_datasets


# --- MAIN FUNCTION ---

def main():
    # datasets_choices = ["mnist", "fashionmnist", "cifar10"]
    datasets_choices = ["cifar10"]
    # epsilon_values = [None, 75.0, 50.0, 25.0, 10.0, 5.0, 1.0, 0.1, 0.01, 0.001]
    epsilon_values = [0.1, 0.01, 0.001]
    num_clients = 25
    rounds = 200
    epochs = 2
    delta = 1e-5

    # Prepare log directory
    log_dir = "./log"
    os.makedirs(log_dir, exist_ok=True)

    # Prepare summary CSV for attack metrics
    summary_csv_path = os.path.join(log_dir, "mia_accuracy_summary.csv")
    with open(summary_csv_path, mode="w", newline="") as summary_file:
        summary_writer = csv.writer(summary_file)
        # Updated header with global model accuracy
        summary_writer.writerow([
            "Dataset", "Epsilon",
            "ART MIA Accuracy", "TPR", "FPR", "Precision",
            "Loss Attack Overall Accuracy", "Avg Member Loss", "Avg Non-Member Loss", "Loss Attack Threshold",
            "Global Model Accuracy"
        ])

        for dataset_choice in datasets_choices:
            print(f"\nProcessing dataset: {dataset_choice}")
            # Load datasets and select appropriate model
            if dataset_choice == "mnist":
                train_dataset, test_dataset = get_mnist_datasets()
                model_fn = mnist_model
            elif dataset_choice == "cifar10":
                train_dataset, test_dataset = get_cifar10_datasets()
                model_fn = cifar10_model
            elif dataset_choice == "fashionmnist":
                train_dataset, test_dataset = get_fashionmnist_datasets()
                model_fn = fashionmnist_model
            else:
                print(f"Unsupported dataset: {dataset_choice}. Skipping...")
                continue

            for epsilon in epsilon_values:
                epsilon_str = "none" if epsilon is None else str(epsilon)
                print(f"\nTraining {dataset_choice} with ε={epsilon_str}")

                # Distribute data among clients
                client_datasets = distribute_data_among_clients(train_dataset, num_clients)

                # Create DP clients
                clients = [
                    Client(model_fn, client_datasets[i], batch_size=32, learning_rate=0.01, device=device, epsilon=epsilon)
                    for i in range(num_clients)
                ]

                # Initialize federated learning instance
                fed_learning = FederatedLearningWithDP(clients, model_fn, dataset_choice, epsilon, delta)

                # Train federated model
                accuracies = fed_learning.train(rounds, epochs, test_dataset)
                # Save final global model accuracy as the last round's accuracy
                global_acc = accuracies[-1]
                # Save global model
                model_save_path = os.path.join(log_dir, f"{dataset_choice}_global_model_epsilon_{epsilon_str}.pt")
                torch.save(fed_learning.global_model.state_dict(), model_save_path)
                print(f"Global model saved at {model_save_path}")

                # Create a shadow model (could be a fresh instantiation)
                shadow_model = model_fn().to(device)

                # Prepare DataLoaders for MIA (using 1000 samples from train and test)
                member_size = 1000
                non_member_size = 1000
                member_indices = list(range(member_size))
                non_member_indices = list(range(member_size, member_size + non_member_size))

                member_subset = Subset(train_dataset, member_indices)
                non_member_subset = Subset(test_dataset, non_member_indices)
                member_loader = DataLoader(member_subset, batch_size=32, shuffle=True)
                non_member_loader = DataLoader(non_member_subset, batch_size=32, shuffle=True)

                # Run ART-based membership inference attack
                mia = MembershipInferenceAttack(shadow_model, fed_learning.global_model, device)
                mia.train_attack_model(member_loader, non_member_loader)
                art_accuracy, tpr, fpr, precision = mia.evaluate_attack(member_loader, non_member_loader)
                print(f"ART MIA Accuracy with ε={epsilon_str}: {art_accuracy:.2f}% | TPR: {tpr:.3f} | FPR: {fpr:.3f} | Precision: {precision:.3f}")

                # Run loss-based membership inference attack
                loss_attack_accuracy, avg_member_loss, avg_non_member_loss, loss_threshold = run_loss_based_attack(
                    fed_learning.global_model, member_loader, non_member_loader, device)
                print(f"Loss-based MIA Accuracy with ε={epsilon_str}: {loss_attack_accuracy:.2f}%")
                print(f"Avg Member Loss: {avg_member_loss:.4f}, Avg Non-member Loss: {avg_non_member_loss:.4f}, Loss Threshold: {loss_threshold:.4f}")

                # Log both ART and loss-based attack metrics along with global accuracy to CSV
                summary_writer.writerow([
                    dataset_choice, epsilon_str,
                    f"{art_accuracy:.2f}", f"{tpr:.3f}", f"{fpr:.3f}", f"{precision:.3f}",
                    f"{loss_attack_accuracy:.2f}", f"{avg_member_loss:.4f}", f"{avg_non_member_loss:.4f}", f"{loss_threshold:.4f}",
                    f"{global_acc:.2f}"
                ])

                # Save accuracy per round to separate CSV
                accuracy_csv_filename = os.path.join(log_dir, f"{dataset_choice}_accuracy_epsilon_{epsilon_str}.csv")
                with open(accuracy_csv_filename, mode="w", newline="") as acc_file:
                    writer = csv.writer(acc_file)
                    writer.writerow(["Round", "Accuracy"])
                    for round_num, acc in enumerate(accuracies, start=1):
                        writer.writerow([round_num, acc])

                # Plot accuracy vs rounds
                plt.figure(figsize=(10, 6))
                plt.plot(range(1, rounds + 1), accuracies, label=f"ε = {epsilon_str}")
                plt.xlabel("Training Rounds")
                plt.ylabel("Global Accuracy (%)")
                plt.title(f"Global Model Accuracy vs Rounds ({dataset_choice}, ε = {epsilon_str})")
                plt.legend()
                plt.grid(True)
                plot_filename = os.path.join(log_dir, f"{dataset_choice}_accuracy_vs_rounds_epsilon_{epsilon_str}.png")
                plt.savefig(plot_filename)
                plt.close()
                print(f"Accuracy plot saved as {plot_filename}.")

    print(f"\nSummary of attack metrics saved to {summary_csv_path}.")

if __name__ == "__main__":
    main()