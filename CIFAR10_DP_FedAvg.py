import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

'''
def load_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    return trainset, testset
'''
def partition_data(trainset, num_clients=4):
    client_indices = [list(range(i * len(trainset) // num_clients, (i + 1) * len(trainset) // num_clients)) for i in range(num_clients)]
    client_loaders = [DataLoader(trainset, batch_size=64, sampler=SubsetRandomSampler(indices)) for indices in client_indices]
    return client_loaders

def get_test_loader(testset):
    test_loader = DataLoader(testset, batch_size=64, shuffle=True)
    return test_loader

def load_dataset(dataset_name):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    if dataset_name == 'CIFAR10':
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    # Add more datasets as needed
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return trainset, testset



def train_client(loader, model, epochs, delta, epsilon):
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    for epoch in range(epochs):
        for data, target in loader:
            data, target = data.cpu(), target.cpu()
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
    return model

def aggregate_models(global_model, client_models):
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0).mean(0)
    global_model.load_state_dict(global_dict)
    return global_model

def test_model(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cpu(), target.cpu()
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return 100. * correct / len(test_loader.dataset)
'''
def run_fedavg(num_clients=4, num_rounds=5, local_epochs=1, epsilon_values=[0.002, 1.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0], delta=0.002):
    trainset, testset = load_dataset()
    client_loaders = partition_data(trainset, num_clients)
    test_loader = get_test_loader(testset)
    
    global_model = SimpleCNN().cpu()

    if not os.path.exists('./log'):
        os.makedirs('./log')
    
    for epsilon in epsilon_values:
        dp_accuracies = []
        global_model_dp = SimpleCNN().cpu()

        for round in range(num_rounds):
            client_models = [train_client(loader, global_model_dp, local_epochs, delta, epsilon) for loader in client_loaders]
            global_model_dp = aggregate_models(global_model_dp, client_models)
            accuracy = test_model(global_model_dp, test_loader)
            dp_accuracies.append(accuracy)
            print(f"Round {round + 1}, Epsilon {epsilon}: Global DP model accuracy: {accuracy:.4f}")
        
        # Save accuracy data to .dat file with round numbers
        with open(f'./log/accuracy_epsilon_{epsilon}.dat', 'w') as f:
            for round_number, acc in enumerate(dp_accuracies, start=1):
                f.write(f"{round_number} {acc}\n")

        # Plot accuracy vs. rounds for this epsilon
        plt.figure(figsize=(10, 8))
        plt.plot(range(1, num_rounds + 1), dp_accuracies, label=f'DP-FedAvg ε={epsilon}', marker='o')
        plt.xlabel('Global Round')
        plt.ylabel('Testing Accuracy')
        plt.title(f'DP-FedAvg Accuracy vs Rounds for ε={epsilon} on CIFAR-10')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'./log/DP_FedAvg_epsilon_{epsilon}.png')
        plt.close()

    print("All simulations completed and results saved.")
'''

def run_fedavg(dataset_name='CIFAR10', num_clients=4, num_rounds=150, local_epochs=1, epsilon_values= [0.0, 1.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0], delta=0.002):
    trainset, testset = load_dataset(dataset_name)  # Pass dataset_name to the load_dataset function
    client_loaders = partition_data(trainset, num_clients)
    test_loader = get_test_loader(testset)

    global_model = SimpleCNN().cpu()

    if not os.path.exists('./log'):
        os.makedirs('./log')

    for epsilon in epsilon_values:
        dp_accuracies = []
        global_model_dp = SimpleCNN().cpu()

        for round in range(num_rounds):
            client_models = [train_client(loader, global_model_dp, local_epochs, delta, epsilon) for loader in client_loaders]
            global_model_dp = aggregate_models(global_model_dp, client_models)
            accuracy = test_model(global_model_dp, test_loader)
            dp_accuracies.append(accuracy)
            print(f"Round {round + 1}, Epsilon {epsilon}: Global DP model accuracy: {accuracy:.4f}")
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

    print("All simulations completed and results saved.")

if __name__ == "__main__":
    dataset_name = 'CIFAR10'  # You can change this to other datasets as needed, like 'MNIST'
    run_fedavg(dataset_name)  # Now passing dataset_name when calling run_fedavg

