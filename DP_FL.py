import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split, Subset
from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant
import matplotlib
import platform
import os
import random
import csv
import matplotlib.pyplot as plt

# Set matplotlib backend explicitly for non-GUI use
matplotlib.use("agg")

# Determine device based on OS and CUDA availability
if platform.system() == "Windows" or platform.system() == "Linux":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
elif platform.system() == "Darwin":  # MacOS
    device = torch.device("mps" if torch.has_mps else "cpu")  # macOS Metal Performance Shaders (MPS)
else:
    device = torch.device("cpu")  # Default to CPU for unknown OS
print(f"Using device: {device}")

# Define MNIST model
def mnist_model():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )

# Federated Learning Client
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

        # Apply differential privacy if epsilon is provided
        if self.epsilon is not None:
            self.privacy_engine = PrivacyEngine()
            self.model, self.optimizer, self.dataloader = self.privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.dataloader,
                noise_multiplier=self._calculate_noise_multiplier(),
                max_grad_norm=1.0,
            )

    def _calculate_noise_multiplier(self):
        if self.epsilon and isinstance(self.epsilon, (float, int)):
            return 1.0 / self.epsilon
        else:
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


# Federated Learning with Differential Privacy
class FederatedLearningWithDP:
    def __init__(self, clients, model, dataset_choice, epsilon, delta):
        self.clients = clients
        self.global_model = model().to(device)
        self.dataset_choice = dataset_choice
        self.epsilon = epsilon
        self.delta = delta
        self.privacy_accountant = RDPAccountant()
        self.noise_multiplier = 1.0 / epsilon if epsilon else None

    def average_weights_with_noise(self, weights_list):
        avg_weights = weights_list[0]
        for key in avg_weights.keys():
            for i in range(1, len(weights_list)):
                avg_weights[key] += weights_list[i][key]
            avg_weights[key] = torch.div(avg_weights[key], len(weights_list))
            if self.epsilon:
                noise_std = 1.0 / self.epsilon
                noise = torch.normal(mean=0, std=noise_std, size=avg_weights[key].size()).to(device)
                avg_weights[key] += noise
        return avg_weights

    def train(self, rounds, epochs):
        global_accuracies = []
        for rnd in range(rounds):
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
                data, target = data.to(device), target.to(device)
                output = self.global_model(data)
                _, predicted = torch.max(output, 1)
                correct += (predicted == target).sum().item()
                total += target.size(0)
        accuracy = 100 * correct / total
        print(f"Global Model Accuracy: {accuracy:.2f}%")
        return accuracy


# Helper functions
def get_mnist_datasets():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    full_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
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


# Main function
def main():
    train_dataset, test_dataset = get_mnist_datasets()
    num_clients = 10
    rounds = 25
    epochs = 1
    epsilon = 1.0
    delta = 1e-5

    client_datasets = distribute_data_among_clients(train_dataset, num_clients)
    clients = [
        Client(mnist_model, client_datasets[i], batch_size=32, learning_rate=0.01, device=device, epsilon=epsilon)
        for i in range(num_clients)
    ]
    fed_learning = FederatedLearningWithDP(clients, mnist_model, "mnist", epsilon, delta)
    accuracies = fed_learning.train(rounds, epochs)

    os.makedirs("./log", exist_ok=True)
    csv_filename = "./log/mnist_accuracy.csv"
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Round", "Accuracy"])
        for round_num, accuracy in enumerate(accuracies, start=1):
            writer.writerow([round_num, accuracy])

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, rounds + 1), accuracies, label=f"ε = {epsilon}")
    plt.xlabel("Training Rounds")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Global Model Accuracy vs Training Rounds (ε = {epsilon})")
    plt.legend()
    plt.grid(True)
    plt.savefig("./log/mnist_accuracy_vs_rounds.png")
    print("Plot saved successfully.")


if __name__ == "__main__":
    main()
