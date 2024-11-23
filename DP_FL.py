import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant
import matplotlib
matplotlib.use("agg")  # Set to a non-GUI backend explicitly
#matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import os
import csv

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        # Validate epsilon and calculate noise multiplier
        if self.epsilon and isinstance(self.epsilon, (float, int)):
            return 1.0 / self.epsilon
        else:
            return None


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

                # Debugging logs
               # print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(self.dataloader)}, Loss: {loss.item()}")

        print("Training complete for this client.")


    def get_weights(self):
        # Handle Opacus wrapper: Use _module's state_dict if wrapped
        if hasattr(self.model, '_module'):
            return self.model._module.state_dict()
        else:
            return self.model.state_dict()

    def set_weights(self, state_dict):
        # Handle Opacus wrapper: Load weights into _module if wrapped
        if hasattr(self.model, '_module'):
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

        # Calculate noise multiplier based on epsilon and delta
        if epsilon and isinstance(epsilon, (float, int)):
            # Define the sampling probability
            sample_rate = 1 / len(clients)  # Assuming equal sampling
            # Calculate the noise multiplier
            self.noise_multiplier = 1.0 / epsilon
        else:
            self.noise_multiplier = None


    def average_weights_with_noise(self, weights_list):
        avg_weights = weights_list[0]
        for key in avg_weights.keys():
            for i in range(1, len(weights_list)):
                avg_weights[key] += weights_list[i][key]
            avg_weights[key] = torch.div(avg_weights[key], len(weights_list))
            # Add noise to aggregated weights if epsilon is valid
            if self.epsilon and isinstance(self.epsilon, (float, int)):
                noise_std = 1.0 / self.epsilon  # Adjust as needed
                noise = torch.normal(mean=0, std=noise_std, size=avg_weights[key].size()).to(device)
                avg_weights[key] += noise
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
            avg_weights = self.average_weights_with_noise(client_weights)
            self.global_model.load_state_dict(avg_weights)
            accuracy = self.evaluate_global_model()
            global_accuracies.append(accuracy)
            
            # Log cumulative privacy budget
            if self.noise_multiplier is not None:
                self.privacy_accountant.step(noise_multiplier=self.noise_multiplier, sample_rate=1 / len(self.clients))
                print(f"Privacy budget spent: ε={self.privacy_accountant.get_epsilon(delta=self.delta):.2f}")
            else:
                print("Noise multiplier is not set. Skipping privacy budget tracking.")
        return global_accuracies


    def evaluate_global_model(self):
        self.global_model.eval()
        test_loader = DataLoader(self.clients[0].dataset, batch_size=32, shuffle=False)
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

# Main function
def main():
    # Define dataset choice and parameters
    dataset_choice = "mnist"  # Use "mnist" for now, you can make it dynamic later
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    
    # Load MNIST dataset
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # Set parameters
    num_clients = 5
    rounds = 5
    epochs = 1
    epsilon = 1.0  # Privacy parameter
    delta = 1e-5

    # Use a clean string for the dataset name
    dataset_name = dataset_choice

    # Create clients
    clients = [Client(mnist_model, dataset, batch_size=32, learning_rate=0.01, device=device, epsilon=epsilon) for _ in range(num_clients)]

    # Federated learning instance
    fed_learning = FederatedLearningWithDP(clients, mnist_model, dataset_name, epsilon, delta)

    # Train federated model
    accuracies = fed_learning.train(rounds, epochs)

    # Save accuracy data
    os.makedirs('./log', exist_ok=True)
    csv_filename = f'./log/{dataset_name}_{epsilon}_accuracy.csv'
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
    plot_filename = f'./log/{dataset_name}_{epsilon}_accuracy_vs_rounds.png'
    plt.savefig(plot_filename)
    print(f"Plot saved successfully as {plot_filename}.")


if __name__ == "__main__":
    main()
