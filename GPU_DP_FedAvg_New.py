import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import seaborn as sns
from opacus import PrivacyEngine

# Check if GPU is available and set the device accordingly
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"Selected GPU: {torch.cuda.get_device_name(device)}")

# Enable cuDNN benchmark for optimized performance on the GPU
torch.backends.cudnn.benchmark = True

class SimpleCNN(nn.Module):
    def __init__(self, input_channels=3, img_size=32):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        if img_size == 32:  # For CIFAR-10
            self.fc1 = nn.Linear(64 * 4 * 4, 256)
        elif img_size == 28:  # For MNIST
            self.fc1 = nn.Linear(64 * 3 * 3, 256)
        else:
            raise ValueError("Unsupported image size")
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def partition_data(trainset, num_clients=4):
    client_indices = [list(range(i * len(trainset) // num_clients, (i + 1) * len(trainset) // num_clients)) for i in range(num_clients)]
    client_loaders = [DataLoader(trainset, batch_size=128, sampler=SubsetRandomSampler(indices), num_workers=4, pin_memory=True) for indices in client_indices]
    return client_loaders

def get_test_loader(testset):
    test_loader = DataLoader(testset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    return test_loader

def load_dataset(dataset_name):
    if dataset_name == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        img_size = 32
    elif dataset_name == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        img_size = 28
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return trainset, testset, img_size

def train_client(loader, model, epochs, delta, epsilon, record_logits=False):
    model.to(device)
    lr = 0.001 if epsilon == 0.0 else 0.0005
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if epsilon > 0.0:
        privacy_engine = PrivacyEngine()
        model, optimizer, loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=loader,
            noise_multiplier=2.0,  # Increased noise multiplier for stronger DP effect
            max_grad_norm=1.0
        )

    train_logits = []
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target).to(device)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if record_logits:
                train_logits.extend([(logit, t) for logit, t in zip(output.detach().cpu().numpy(), target.cpu().numpy())])
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(loader):.4f}")

    # Print effective epsilon for DP training
    if epsilon > 0.0:
        effective_epsilon = privacy_engine.get_epsilon(delta)
        print(f"Effective epsilon after training: {effective_epsilon:.2f}")

    # Delete privacy engine to avoid adding hooks multiple times
    if epsilon > 0.0:
        del privacy_engine

    return model, train_logits if record_logits else model

def aggregate_models(global_model, client_models):
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        if all(k in client_model.state_dict() for client_model in client_models):
            global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0).mean(0)
    global_model.load_state_dict(global_dict)
    return global_model

def test_model(model, test_loader, record_logits=False):
    model.eval()
    model.to(device)
    test_logits = []
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            if record_logits:
                test_logits.extend([(logit, t) for logit, t in zip(output.cpu().numpy(), target.cpu().numpy())])
    return 100. * correct / len(test_loader.dataset), test_logits if record_logits else 100. * correct / len(test_loader.dataset)

def prepare_attack_data(member_logits, non_member_logits):
    member_logits_flat = np.array([logit.flatten() for logit, _ in member_logits], dtype=object)
    non_member_logits_flat = np.array([logit.flatten() for logit, _ in non_member_logits], dtype=object)

    max_length = max(max(len(logit) for logit in member_logits_flat), max(len(logit) for logit in non_member_logits_flat))
    member_logits_padded = np.array([np.pad(logit, (0, max_length - len(logit))) for logit in member_logits_flat])
    non_member_logits_padded = np.array([np.pad(logit, (0, max_length - len(logit))) for logit in non_member_logits_flat])

    X = np.concatenate([member_logits_padded, non_member_logits_padded])
    y = np.concatenate([np.ones(len(member_logits_padded)), np.zeros(len(non_member_logits_padded))])
    return X, y

def train_attack_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    attack_model = LogisticRegression()
    attack_model.fit(X_train, y_train)
    predictions = attack_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Attack Model Accuracy: {accuracy:.4f}")
    return attack_model, accuracy

def run_fedavg(dataset_name, num_clients=4, num_rounds=20, local_epochs=5, epsilon_values=[0.0, 0.02, 1.0, 5.0, 10.0, 25.0, 50.0], delta=0.002):
    trainset, testset, img_size = load_dataset(dataset_name)
    client_loaders = partition_data(trainset, num_clients)
    test_loader = get_test_loader(testset)

    input_channels = 1 if dataset_name == 'MNIST' else 3

    if not os.path.exists('./log'):
        os.makedirs('./log')

    attack_results = []
    global_accuracies = []

    for epsilon in epsilon_values:
        dp_accuracies = []
        global_model_dp = SimpleCNN(input_channels=input_channels, img_size=img_size).to(device)

        for round in range(num_rounds):
            client_models = []

            # Train each client model and collect them for aggregation
            for loader in client_loaders:
                client_model = SimpleCNN(input_channels=input_channels, img_size=img_size).to(device)
                client_model.load_state_dict(global_model_dp.state_dict())
                trained_client_model, _ = train_client(loader, client_model, local_epochs, delta, epsilon)
                client_models.append(trained_client_model)

            # Aggregate client models to update the global model
            global_model_dp = aggregate_models(global_model_dp, client_models)
            accuracy, _ = test_model(global_model_dp, test_loader)
            dp_accuracies.append(accuracy)
            print(f"Round {round + 1}, Epsilon {epsilon}: Global DP model accuracy: {accuracy:.4f}")

        global_accuracies.append((epsilon, dp_accuracies))
        with open(f'./log/{dataset_name}_DP_FedAvg_epsilon_{epsilon}.dat', 'w') as f:
            for round_number, acc in enumerate(dp_accuracies, start=1):
                f.write(f"{round_number} {acc}\n")

        plt.figure(figsize=(10, 8))
        plt.plot(range(1, num_rounds + 1), dp_accuracies, label=f'DP-FedAvg ε={epsilon}', marker='o')
        plt.xlabel('Global Round')
        plt.ylabel('Testing Accuracy')
        plt.title(f'{dataset_name} - DP-FedAvg Accuracy vs Rounds for ε={epsilon}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'./log/{dataset_name}_DP_FedAvg_epsilon_{epsilon}.png')
        plt.close()

    # Create a plot for Global DP Model Accuracy vs Rounds for all Epsilon Values
    plt.figure(figsize=(10, 8))
    for epsilon, accuracies in global_accuracies:
        plt.plot(range(1, num_rounds + 1), accuracies, label=f'ε={epsilon}', marker='o')
    plt.xlabel('Global Round')
    plt.ylabel('Testing Accuracy')
    plt.title(f'{dataset_name} - DP-FedAvg Accuracy vs Rounds')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./log/{dataset_name}_accuracy_vs_rounds.png')
    plt.close()

    print("All simulations completed and results saved.")


if __name__ == "__main__":
    dataset_name = 'MNIST'
    run_fedavg(dataset_name)
