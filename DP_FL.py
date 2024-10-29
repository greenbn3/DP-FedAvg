# Import required libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import random_split
from opacus import PrivacyEngine  # Import Opacus for differential privacy
import os
import warnings
import signal
import csv

from dataset import FEMNIST, ShakeSpeare  # Import custom datasets

# Suppress Opacus warnings
warnings.filterwarnings("ignore", category=UserWarning, module="opacus")

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

# Fashion-MNIST Model (Same as MNIST)
def fashion_mnist_model():
    return mnist_model()

# Shakespeare Model (Assuming a simple LSTM-based model)
def shakespeare_model():
    class LSTMModel(nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
            super(LSTMModel, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)
        
        def forward(self, x):
            embedded = self.embedding(x)
            lstm_out, _ = self.lstm(embedded)
            out = self.fc(lstm_out[:, -1, :])  # Use the last output
            return out

    # Parameters should be adjusted based on your dataset
    vocab_size = 100  # Example value
    embedding_dim = 128
    hidden_dim = 256
    output_dim = 26  # Assuming 26 letters
    return LSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim)

# Define a function to get dataset and model based on choice
def get_dataset_and_model(dataset_choice):
    if dataset_choice == "mnist":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        model = mnist_model
    elif dataset_choice == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        model = cifar10_model
    elif dataset_choice == "fashion-mnist":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset = datasets.FashionMNIST(root='./data/fashion-mnist', train=True, download=True, transform=transform)
        model = fashion_mnist_model
    elif dataset_choice == "femnist":
        dataset = FEMNIST(train=True, transform=transforms.ToTensor())
        model = mnist_model  # Assuming the same model as MNIST
    elif dataset_choice == "shakespeare":
        dataset = ShakeSpeare(train=True)
        model = shakespeare_model
    else:
        raise ValueError("Invalid dataset choice.")
    return dataset, model

# Handle interrupt signal to reset GPU resources
def handle_interrupt(signal, frame):
    print("\nInterrupt received, resetting GPU resources...")
    torch.cuda.empty_cache()
    exit(0)

signal.signal(signal.SIGINT, handle_interrupt)

# Federated learning client class
class Client:
    def __init__(self, model, dataset, batch_size, learning_rate, device, epsilon=None, delta=1e-5):
        self.model = model().to(device)
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.loss_fn = nn.CrossEntropyLoss() if dataset != "shakespeare" else nn.MSELoss()
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
                # Freeing up memory to prevent memory leaks
                del data, target, output, loss
                torch.cuda.empty_cache()

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
    def __init__(self, clients, model, dataset_choice):
        self.clients = clients
        self.global_model = model().to(device)
        self.dataset_choice = dataset_choice  # Store dataset_choice as an instance variable

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
        test_loader = DataLoader(self.clients[0].dataset, batch_size=32, shuffle=False)
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = self.global_model(data)
                if self.dataset_choice == "shakespeare":  # Use the instance variable
                    # Assuming one-hot encoding for targets
                    _, predicted = torch.max(output, 1)
                    target_labels = torch.argmax(target, dim=1)
                    correct += (predicted == target_labels).sum().item()
                else:
                    _, predicted = torch.max(output, 1)
                    correct += (predicted == target).sum().item()
                total += target.size(0)
        accuracy = 100 * correct / total
        print(f"Global Model Accuracy: {accuracy:.2f}%")
        return accuracy

# Main function
def main():
    dataset_choice = "mnist"  # Change this to "mnist", "cifar10", "fashion-mnist", "femnist", or "shakespeare"
    
    dataset, model_type = get_dataset_and_model(dataset_choice)

    num_clients = 2
    rounds = 100
    epochs = 1
    epsilon = 0.01
    # Uncomment below lines to take inputs from the user
    # dataset_choice = input("Choose dataset (mnist/cifar10/fashion-mnist/femnist/shakespeare): ").strip().lower()
    # num_clients = int(input("Enter number of clients: "))
    # rounds = int(input("Enter number of training rounds: "))
    # epochs = int(input("Enter number of epochs per round: "))
    # epsilon = input("Enter privacy epsilon value (or 'none' for no privacy): ").strip().lower()
    # epsilon = float(epsilon) if epsilon != 'none' else None

    # Create clients
    clients = [Client(model_type, dataset, batch_size=32, learning_rate=0.01, device=device, epsilon=epsilon) for _ in range(num_clients)]
    
    # Create Federated Learning instance
    fed_learning = FederatedLearning(clients, model_type, dataset_choice)  # Pass dataset_choice

    # Train Federated Model
    accuracies = fed_learning.train(rounds, epochs)

    # Create log directory if it doesn't exist
    os.makedirs('./log', exist_ok=True)

    # Save accuracy data to CSV
    csv_filename = f'./log/{dataset_choice}_{epsilon}_accuracy_data.csv'
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Round', 'Accuracy'])
        for round_num, accuracy in enumerate(accuracies, start=1):
            writer.writerow([round_num, accuracy])

    # Plot Accuracy vs Training Rounds
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, rounds + 1), accuracies, label=f'ε = {epsilon}')
    plt.xlabel('Training Rounds')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Global Model Accuracy vs Training Rounds (ε = {epsilon})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./log/{dataset_choice}_{epsilon}_accuracy_vs_rounds.png')
    plt.show()

if __name__ == "__main__":
    main()
