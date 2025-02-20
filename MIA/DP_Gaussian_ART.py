import os
import random
import platform
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms

from opacus import PrivacyEngine
from art.estimators.classification import PyTorchClassifier
from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
import numpy as np
import csv
import matplotlib.pyplot as plt

# 1. DEVICE SELECTION -----------------------------------------------------
if platform.system() in ["Windows", "Linux"]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
elif platform.system() == "Darwin":  # macOS
    # Use the originally selected device (e.g. mps if available)
    device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# 2. MODEL DEFINITIONS ----------------------------------------------------
def mnist_model():
    """Simple feedforward network for MNIST (1,28,28)"""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )

def cifar10_model():
    """Simple CNN for CIFAR-10 (3,32,32)"""
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),   # 16x16 out
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),   # 8x8 out
        nn.Flatten(),
        nn.Linear(64 * 8 * 8, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )

# 3. CLIENT CLASS (WITH OPACUS DP) ---------------------------------------
class Client:
    def __init__(self, model_fn, dataset, batch_size, learning_rate, device, epsilon=None, delta=1e-5):
        self.model = model_fn().to(device)
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        self.device = device
        self.epsilon = epsilon
        self.delta = delta

        # If epsilon is provided (and not "none"), add DP using the Gaussian mechanism.
        if self.epsilon and self.epsilon != "none":
            noise_multiplier = 1.0 / float(self.epsilon)
            self.privacy_engine = PrivacyEngine()
            self.model, self.optimizer, self.dataloader = self.privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.dataloader,
                noise_multiplier=noise_multiplier,
                max_grad_norm=1.0
            )

    def train(self, epochs):
        self.model.train()
        for _ in range(epochs):
            for data, target in self.dataloader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.loss_fn(outputs, target)
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

# 4. FEDERATED LEARNING CLASS (NO EXTRA NOISE AT SERVER) -----------------
class FederatedLearningNoServerNoise:
    def __init__(self, clients, model_fn, epsilon, delta):
        self.clients = clients
        self.global_model = model_fn().to(device)
        self.epsilon = epsilon
        self.delta = delta
        self.device = device

    def train(self, rounds, epochs, test_dataset):
        accuracies = []
        for r in range(rounds):
            print(f"Round {r+1}/{rounds}")
            client_weights = []
            for client in self.clients:
                client.set_weights(self.global_model.state_dict())
                client.train(epochs)
                client_weights.append(client.get_weights())
            new_global_state_dict = self.average_weights(client_weights)
            self.global_model.load_state_dict(new_global_state_dict)
            acc = self.evaluate_global_model(test_dataset)
            accuracies.append(acc)
        return accuracies

    def average_weights(self, weights_list):
        avg_weights = {}
        for key in weights_list[0].keys():
            avg_weights[key] = torch.stack([w[key] for w in weights_list], dim=0).mean(dim=0)
        return avg_weights

    def evaluate_global_model(self, test_dataset):
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        self.global_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.global_model(data)
                _, predicted = torch.max(outputs, dim=1)
                correct += (predicted == target).sum().item()
                total += len(target)
        acc = 100.0 * correct / total
        print(f"Global Model Test Accuracy: {acc:.2f}%")
        return acc

# 5. MIA ATTACK WITH ART (Black-Box) -------------------------------------
def run_membership_inference_attack_art(global_model, mia_in_dataset, mia_out_dataset, device):
    """
    Runs a black-box membership inference attack using ART on disjoint MIA datasets:
      - mia_in_dataset:  Members (hold-out from training)
      - mia_out_dataset: Non-members (hold-out from test)
    """
    in_loader = DataLoader(mia_in_dataset, batch_size=32, shuffle=False)
    out_loader = DataLoader(mia_out_dataset, batch_size=32, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(global_model.parameters(), lr=0.01)

    for x_batch, _ in in_loader:
        input_shape = x_batch.shape[1:]
        break

    art_classifier = PyTorchClassifier(
        model=global_model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=input_shape,
        nb_classes=10,
        device_type="gpu" if "cuda" in device.type else "cpu",
    )

    def loader_to_numpy(data_loader):
        xs, ys = [], []
        for x, y in data_loader:
            # Convert tensors to numpy arrays (move to CPU only for conversion)
            xs.append(x.to("cpu").numpy())
            ys.append(y.to("cpu").numpy())
        X = np.concatenate(xs, axis=0)
        Y = np.concatenate(ys, axis=0)
        return X, Y

    member_x, member_y = loader_to_numpy(in_loader)
    non_member_x, non_member_y = loader_to_numpy(out_loader)

    mia_attack = MembershipInferenceBlackBox(art_classifier, attack_model_type="nn")

    mia_attack.fit(
        x=member_x,
        y=member_y,
        membership=np.ones(len(member_x)),
        test_x=non_member_x,
        test_y=non_member_y,
        test_membership=np.zeros(len(non_member_x))
    )

    inferred_in = mia_attack.infer(member_x, member_y)
    inferred_out = mia_attack.infer(non_member_x, non_member_y)

    correct_in = (inferred_in == 1).sum()
    correct_out = (inferred_out == 0).sum()
    total_in = len(inferred_in)
    total_out = len(inferred_out)

    in_acc = correct_in / total_in
    out_acc = correct_out / total_out
    overall_acc = (correct_in + correct_out) / (total_in + total_out)

    print(f"[ART MIA] In-Set Accuracy  : {in_acc*100:.2f}% ({correct_in}/{total_in})")
    print(f"[ART MIA] Out-of-Set Acc   : {out_acc*100:.2f}% ({correct_out}/{total_out})")
    print(f"[ART MIA] Overall Acc      : {overall_acc*100:.2f}%")
    return in_acc, out_acc, overall_acc

# 6. PLOTTING MODEL OUTPUT DISTRIBUTIONS -------------------------------
def plot_model_outputs(model, dataset, device, title, save_path):
    """
    Plots histograms of the maximum softmax probabilities and maximum logit values
    for the given dataset.
    """
    model = model.to('cpu')
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    max_softmax = []
    max_logits = []
    with torch.no_grad():
        for data, _ in loader:
            data = data.to('cpu')
            logits = model(data)
            softmax = torch.softmax(logits, dim=1)
            max_softmax.extend(softmax.max(dim=1)[0].cpu().numpy())
            max_logits.extend(logits.max(dim=1)[0].cpu().numpy())
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(max_softmax, bins=50, color='blue', alpha=0.7)
    plt.title(f"{title} - Max Softmax")
    plt.xlabel("Max Softmax Value")
    plt.ylabel("Frequency")
    plt.subplot(1, 2, 2)
    plt.hist(max_logits, bins=50, color='green', alpha=0.7)
    plt.title(f"{title} - Max Logit")
    plt.xlabel("Max Logit Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

# 7. LOGGING -------------------------------------------------------------
def log_training_progress(dataset_name, epsilon_str, accuracies, log_dir="./gaussian_log"):
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, f"{dataset_name}_training_progress_epsilon_{epsilon_str}.csv")
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Round", "Accuracy"])
        for r_idx, acc in enumerate(accuracies, start=1):
            writer.writerow([r_idx, acc])
    print(f"[LOG] Training progress saved to: {csv_path}")

def log_mia_results(dataset_name, epsilon_str, final_acc, in_acc, out_acc, overall_acc, log_dir="./gaussian_log"):
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, f"{dataset_name}_mia_summary.csv")
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Dataset", "Epsilon", "FinalGlobalAccuracy", "MIA_InSet_Accuracy", "MIA_OutSet_Accuracy", "MIA_Overall_Accuracy"])
        writer.writerow([dataset_name, epsilon_str, f"{final_acc:.2f}", f"{in_acc*100:.2f}", f"{out_acc*100:.2f}", f"{overall_acc*100:.2f}"])
    print(f"[LOG] MIA results appended to: {csv_path}")

# 8. DATASET HELPERS -----------------------------------------------------
def get_mnist_datasets():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    full_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    return full_train, test_data

def get_cifar10_datasets():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # normalize to ~[-1..1]
    ])
    full_train = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    return full_train, test_data

def distribute_data_among_clients(train_dataset, num_clients):
    client_datasets = []
    client_size = len(train_dataset) // num_clients
    indices = list(range(len(train_dataset)))
    random.shuffle(indices)
    for i in range(num_clients):
        client_indices = indices[i * client_size:(i + 1) * client_size]
        client_datasets.append(Subset(train_dataset, client_indices))
    return client_datasets

# 9. MAIN ----------------------------------------------------------------
def main():
    # Choose "mnist" or "cifar10"
    dataset_choice = "cifar10"
    num_clients = 3
    rounds = 150   # Increased training rounds
    epochs = 2
    # Use a more moderate range of epsilon values
    epsilons = [None, 0.5, 1.0, 2.0, 5.0, 10.0]
    delta = 1e-5

    # 1) Load dataset / model based on dataset_choice
    if dataset_choice == "mnist":
        full_train, full_test = get_mnist_datasets()
        model_fn = mnist_model
    elif dataset_choice == "cifar10":
        full_train, full_test = get_cifar10_datasets()
        model_fn = cifar10_model
    else:
        raise ValueError(f"Unknown dataset_choice: {dataset_choice}")

    # 2) Split full datasets into training for federated learning and hold-out sets for MIA
    mia_in_size = 1000   # Members: hold-out from full training set
    mia_out_size = 1000  # Non-members: hold-out from full test set
    train_dataset_for_fed, mia_in_dataset = random_split(
        full_train,
        [len(full_train) - mia_in_size, mia_in_size],
        generator=torch.Generator().manual_seed(42)
    )
    test_dataset_for_fed, mia_out_dataset = random_split(
        full_test,
        [len(full_test) - mia_out_size, mia_out_size],
        generator=torch.Generator().manual_seed(42)
    )

    # 3) Loop over epsilon values
    for eps in epsilons:
        # Distribute the training set (for federated learning) among clients
        client_datasets = distribute_data_among_clients(train_dataset_for_fed, num_clients)
        # Create DP clients (if eps is provided) using the Gaussian PrivacyEngine
        clients = []
        for i in range(num_clients):
            c = Client(
                model_fn=model_fn,
                dataset=client_datasets[i],
                batch_size=32,
                learning_rate=0.01,
                device=device,
                epsilon=eps,
                delta=delta
            )
            clients.append(c)
        # Create the federated aggregator
        fed_aggregator = FederatedLearningNoServerNoise(clients, model_fn, eps, delta)
        
        epsilon_str = "none" if eps is None or eps == "none" else str(eps)
        print(f"\n=== Starting Federated Training ({dataset_choice}) with epsilon={epsilon_str} ===")
        global_accuracies = fed_aggregator.train(rounds, epochs, test_dataset_for_fed)
        print("Federated Training Done.")
        final_global_model = fed_aggregator.global_model
        final_accuracy = global_accuracies[-1] if global_accuracies else 0.0

        # Log training progress
        log_training_progress(
            dataset_name=dataset_choice,
            epsilon_str=epsilon_str,
            accuracies=global_accuracies,
            log_dir="./gaussian_log"
        )

        # 4) Run Membership Inference Attack using the hold-out sets (using the original device)
        print("\nRunning Membership Inference Attack (ART Black-Box)...")
        in_acc, out_acc, overall_acc = run_membership_inference_attack_art(
            final_global_model, mia_in_dataset, mia_out_dataset, device
        )
        print(f"MIA In-Set Accuracy: {in_acc*100:.2f}%")
        print(f"MIA Out-of-Set Accuracy: {out_acc*100:.2f}%")
        print(f"MIA Overall Accuracy: {overall_acc*100:.2f}%")
        log_mia_results(
            dataset_name=dataset_choice,
            epsilon_str=epsilon_str,
            final_acc=final_accuracy,
            in_acc=in_acc,
            out_acc=out_acc,
            overall_acc=overall_acc,
            log_dir="./gaussian_log"
        )
        print(f"Final MIA Accuracy (overall): {overall_acc*100:.2f}%")
        
        # IMPORTANT: Move model back to the original device (e.g. MPS) before plotting
        final_global_model = final_global_model.to(device)
        
        # 5) Plot output distributions for members and non-members using the original device
        os.makedirs("./plots", exist_ok=True)
        plot_model_outputs(final_global_model, mia_in_dataset, device,
                           f"{dataset_choice} Members (ε={epsilon_str})", 
                           f"./plots/{dataset_choice}_members_epsilon_{epsilon_str}.png")
        plot_model_outputs(final_global_model, mia_out_dataset, device,
                           f"{dataset_choice} Non-members (ε={epsilon_str})", 
                           f"./plots/{dataset_choice}_nonmembers_epsilon_{epsilon_str}.png")

if __name__ == "__main__":
    main()