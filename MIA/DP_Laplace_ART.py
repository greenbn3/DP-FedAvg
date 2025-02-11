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

# 1. DEVICE SELECTION -----------------------------------------------------

if platform.system() == "Windows" or platform.system() == "Linux":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
elif platform.system() == "Darwin":  # MacOS
    device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# 2. MODEL DEFINITIONS ----------------------------------------------------

def mnist_model():
    """
    Simple feedforward network for MNIST (1,28,28)
    """
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )

def cifar10_model():
    """
    Simple CNN for CIFAR-10 (3,32,32)
    """
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

# 3. CUSTOM LAPLACE PRIVACY ENGINE ---------------------------------------
class LaplacePrivacyEngine(PrivacyEngine):
    """
    A custom privacy engine that adds Laplace noise calibrated in a
    variance-matched way to the Gaussian mechanism.
    
    With sensitivity Δ = 1 (i.e. max_grad_norm = 1), the standard Gaussian
    mechanism uses: 
       σ = sqrt(2 * ln(1.25/δ)) / ε.
    
    To match variance, we set the Laplace scale to:
       b = sqrt(ln(1.25/δ)) / ε
    (since Var(Laplace(0,b)) = 2b^2).
    
    You can adjust this if you prefer canonical Laplace (i.e. b = 1/ε).
    """
    def __init__(self, epsilon, delta, accountant="rdp", secure_mode=False):
        self.epsilon = epsilon
        self.delta = delta
        super().__init__(accountant=accountant, secure_mode=secure_mode)
    
    def _generate_noise(self, noise_shape):
        with torch.no_grad():
            # If privacy is not applied, return zeros.
            if self.epsilon is None or self.epsilon == 0 or self.max_grad_norm == 0:
                return torch.zeros(noise_shape, device=self.device, dtype=self.dtype)
            # Use variance-matched calibration:
            scale = math.sqrt(math.log(1.25/self.delta)) / float(self.epsilon) * self.max_grad_norm
            laplace_dist = torch.distributions.Laplace(
                loc=torch.tensor(0.0, device=self.device, dtype=self.dtype),
                scale=torch.tensor(scale, device=self.device, dtype=self.dtype)
            )
            return laplace_dist.sample(noise_shape)

# 4. CLIENT CLASS (WITH LAPACE DP) ---------------------------------------
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

        # If epsilon is provided (and not "none"), apply DP using our Laplace mechanism.
        if self.epsilon and self.epsilon != "none":
            # Instead of computing noise_multiplier = 1/ε,
            # we use our custom LaplacePrivacyEngine which computes noise based on:
            #   scale = sqrt(ln(1.25/δ))/ε * max_grad_norm.
            self.privacy_engine = LaplacePrivacyEngine(epsilon=self.epsilon, delta=self.delta, accountant="rdp")
            # We pass a dummy noise_multiplier (e.g. 0.0) since our _generate_noise overrides it.
            self.model, self.optimizer, self.dataloader = self.privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.dataloader,
                noise_multiplier=0.0,  # value is ignored in our custom _generate_noise
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

# 5. FEDERATED LEARNING CLASS (NO EXTRA NOISE AT SERVER) -----------------
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

# 6. MIA ATTACK WITH ART (Black-Box) -------------------------------------
def run_membership_inference_attack_art(global_model, train_dataset, test_dataset, device):
    """
    Demonstrates how to run a black-box membership inference attack 
    from the Adversarial Robustness Toolbox (ART).
    """
    # For demonstration, we choose 1000 "members" from train_dataset & 1000 "non-members" from test_dataset
    member_size = 1000
    non_member_size = 1000

    # Subset for members
    train_subset, _ = random_split(train_dataset, [member_size, len(train_dataset) - member_size])
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=False)

    # Subset for non-members
    test_subset, _ = random_split(test_dataset, [non_member_size, len(test_dataset) - non_member_size])
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)

    # Convert the global model to an ART classifier
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(global_model.parameters(), lr=0.01)

    # Figure out input shape from the first batch
    for x_batch, _ in train_loader:
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

    mia_attack = MembershipInferenceBlackBox(
        art_classifier,
        attack_model_type="nn"
    )

    def loader_to_numpy(data_loader):
        xs, ys = [], []
        for x, y in data_loader:
            xs.append(x.cpu().numpy())
            ys.append(y.cpu().numpy())
        X = np.concatenate(xs, axis=0)
        Y = np.concatenate(ys, axis=0)
        return X, Y

    member_x, member_y = loader_to_numpy(train_loader)
    non_member_x, non_member_y = loader_to_numpy(test_loader)

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
    print(f"[ART MIA] Out-of-Set Acc : {out_acc*100:.2f}% ({correct_out}/{total_out})")
    print(f"[ART MIA] Overall Acc    : {overall_acc*100:.2f}%")

    return in_acc, out_acc, overall_acc

# 7. LOGGING -------------------------------------------------------------
def log_training_progress(dataset_name, epsilon_str, accuracies, log_dir="./laplace_log"):
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, f"{dataset_name}_training_progress_epsilon_{epsilon_str}.csv")
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Round", "Accuracy"])
        for r_idx, acc in enumerate(accuracies, start=1):
            writer.writerow([r_idx, acc])
    print(f"[LOG] Training progress saved to: {csv_path}")

def log_mia_results(dataset_name, epsilon_str, final_acc, in_acc, out_acc, overall_acc, log_dir="./laplace_log"):
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, f"{dataset_name}_mia_summary.csv")
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Dataset", 
                "Epsilon", 
                "FinalGlobalAccuracy", 
                "MIA_InSet_Accuracy", 
                "MIA_OutSet_Accuracy",
                "MIA_Overall_Accuracy"
            ])
        writer.writerow([
            dataset_name,
            epsilon_str,
            f"{final_acc:.2f}",
            f"{in_acc*100:.2f}",
            f"{out_acc*100:.2f}",
            f"{overall_acc*100:.2f}"
        ])
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
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
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
    # Pick "mnist" or "cifar10"
    dataset_choice = "cifar10"
    num_clients = 3
    rounds = 20
    epochs = 2
    epsilons = [None, 0.01, 0.1, 0.25, 0.5, 1.0, 5.0]
    delta = 1e-5
    
    # 1) Load dataset / model based on dataset_choice
    if dataset_choice == "mnist":
        train_dataset, test_dataset = get_mnist_datasets()
        model_fn = mnist_model
    elif dataset_choice == "cifar10":
        train_dataset, test_dataset = get_cifar10_datasets()
        model_fn = cifar10_model
    else:
        raise ValueError(f"Unknown dataset_choice: {dataset_choice}")

    for eps in epsilons:
        # 2) Distribute data among clients
        client_datasets = distribute_data_among_clients(train_dataset, num_clients)

        # 3) Create DP clients (using our Laplace mechanism)
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
        
        # 4) Create aggregator
        fed_aggregator = FederatedLearningNoServerNoise(clients, model_fn, eps, delta)
        
        print(f"\n=== Starting Federated Training ({dataset_choice}) with epsilon={eps} ===")
        global_accuracies = fed_aggregator.train(rounds, epochs, test_dataset)
        print("Federated Training Done.")
        
        final_global_model = fed_aggregator.global_model
        final_accuracy = global_accuracies[-1] if global_accuracies else 0.0
        
        epsilon_str = "none" if eps in [None, "none"] else str(eps)

        # 6) Log training progress
        log_training_progress(
            dataset_name=dataset_choice,
            epsilon_str=epsilon_str,
            accuracies=global_accuracies,
            log_dir="./laplace_log"
        )
        
        # 7) MIA Attack
        print("\nRunning Membership Inference Attack (ART Black-Box)...")
        in_acc, out_acc, overall_acc = run_membership_inference_attack_art(
            final_global_model, train_dataset, test_dataset, device
        )
        
        print(f"MIA In-Set Accuracy: {in_acc*100:.2f}%")
        print(f"MIA Out-of-Set Accuracy: {out_acc*100:.2f}%")
        print(f"MIA Overall Accuracy: {overall_acc*100:.2f}%")
        
        # 8) Log MIA results
        log_mia_results(
            dataset_name=dataset_choice,
            epsilon_str=epsilon_str,
            final_acc=final_accuracy,
            in_acc=in_acc,
            out_acc=out_acc,
            overall_acc=overall_acc,
            log_dir="./laplace_log"
        )
        
        print(f"Final MIA Accuracy (overall): {overall_acc*100:.2f}%")

if __name__ == "__main__":
    main()
