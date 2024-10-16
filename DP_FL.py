# Import required libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import random_split
from opacus import PrivacyEngine  # Import Opacus for differential privacy

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# MNIST Model
def mnist_model():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )

# CIFAR10 Model
def cifar10_model():
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(64 * 8 * 8, 512),
        nn.ReLU(),
        nn.Linear(512, 10),
    )

# Load datasets
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
cifar10_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Federated learning client class
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
        if self.epsilon is not None:
            return 1.0 / self.epsilon  # Adjust this calculation as needed
        return 0.0

    def train(self, epochs):
        self.model.train()
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(self.dataloader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_fn(output, target)
                loss.backward()
                self.optimizer.step()

    def get_weights(self):
        if hasattr(self.model, '_module'):
            # Return state_dict from the underlying model
            return self.model._module.state_dict()
        else:
            return self.model.state_dict()

    def set_weights(self, state_dict):
        if hasattr(self.model, '_module'):
            # Load state_dict into the underlying model
            self.model._module.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)



# Federated Learning Class
class FederatedLearning:
    def __init__(self, clients, model_type):
        self.clients = clients
        self.global_model = model_type().to(device)

    def average_weights(self, weights_list):
        avg_weights = weights_list[0]
        for key in avg_weights.keys():
            for i in range(1, len(weights_list)):
                avg_weights[key] += weights_list[i][key]
            avg_weights[key] = torch.div(avg_weights[key], len(weights_list))
        return avg_weights

    def train(self, rounds, epochs):
        global_accuracies = []
        for rnd in range(rounds):
            print(f"Round {rnd+1}/{rounds}")
            client_weights = []
            for client in self.clients:
                client.set_weights(self.global_model.state_dict())
                client.train(epochs)
                client_weights.append(client.get_weights())
            avg_weights = self.average_weights(client_weights)
            self.global_model.load_state_dict(avg_weights)
            accuracy = self.evaluate_global_model()
            global_accuracies.append(accuracy)
        return global_accuracies

    def evaluate_global_model(self):
        self.global_model.eval()
        test_loader = DataLoader(self.clients[0].dataset, batch_size=64, shuffle=False)
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = self.global_model(data)
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        accuracy = 100 * correct / total
        print(f"Global Model Accuracy: {accuracy:.2f}%")
        return accuracy

# Main function
def main():
    dataset_choice = input("Choose dataset (mnist/cifar10): ").strip().lower()
    if dataset_choice == "mnist":
        dataset = mnist_dataset
        model_type = mnist_model
    elif dataset_choice == "cifar10":
        dataset = cifar10_dataset
        model_type = cifar10_model
    else:
        print("Invalid dataset choice.")
        return

    num_clients = int(input("Enter number of clients: "))
    rounds = int(input("Enter number of training rounds: "))
    epochs = int(input("Enter number of epochs per round: "))
    epsilon = input("Enter privacy epsilon value (or 'none' for no privacy): ").strip().lower()
    epsilon = float(epsilon) if epsilon != 'none' else None

    # Create clients
    clients = [Client(model_type, dataset, batch_size=32, learning_rate=0.01, device=device, epsilon=epsilon) for _ in range(num_clients)]
    
    # Create Federated Learning instance
    fed_learning = FederatedLearning(clients, model_type)

    # Train Federated Model
    accuracies = fed_learning.train(rounds, epochs)

    # Plot Accuracy vs Training Rounds
    plt.plot(range(1, rounds + 1), accuracies)
    plt.xlabel('Training Rounds')
    plt.ylabel('Accuracy (%)')
    plt.title('Global Model Accuracy vs Training Rounds')
    plt.show()

if __name__ == "__main__":
    main()