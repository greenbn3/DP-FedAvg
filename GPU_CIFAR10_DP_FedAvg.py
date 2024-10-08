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

# Check if GPU is available and set the device accordingly
# Explicitly set the device to GPU 0 (NVIDIA GeForce RTX 4060)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Selected GPU: {torch.cuda.get_device_name(device)}")

class SimpleCNN(nn.Module):
    def __init__(self, input_channels=3):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4 if input_channels == 1 else 16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
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
    elif dataset_name == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return trainset, testset

def train_client(loader, model, epochs, delta, epsilon, record_logits=False):
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    train_logits = []
    model.train()
    for epoch in range(epochs):
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            if record_logits:
                train_logits.extend([(logit, t) for logit, t in zip(output.detach().cpu().numpy(), target.cpu().numpy())])

       # print(f"Epoch {epoch + 1}: Memory allocated: {torch.cuda.memory_allocated(device)} bytes")
       # print(f"Epoch {epoch + 1}: Memory cached: {torch.cuda.memory_reserved(device)} bytes")

    return model, train_logits if record_logits else model

def aggregate_models(global_model, client_models):
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0).mean(0)
    global_model.load_state_dict(global_dict)
    return global_model

def test_model(model, test_loader, record_logits=False):
    model.eval()
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
    # Flatten logits if necessary to ensure consistent shape
    member_logits_flat = np.array([logit.flatten() for logit, _ in member_logits], dtype=object)
    non_member_logits_flat = np.array([logit.flatten() for logit, _ in non_member_logits], dtype=object)

    # Pad sequences to ensure consistent shape
    max_length = max(max(len(logit) for logit in member_logits_flat), max(len(logit) for logit in non_member_logits_flat))
    member_logits_padded = np.array([np.pad(logit, (0, max_length - len(logit))) for logit in member_logits_flat])
    non_member_logits_padded = np.array([np.pad(logit, (0, max_length - len(logit))) for logit in non_member_logits_flat])

    # Concatenate member and non-member logits
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

def run_fedavg(dataset_name='MNIST', num_clients=4, num_rounds=10, local_epochs=1, epsilon_values=[0.0, 1.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 1000.0], delta=0.002):
    trainset, testset = load_dataset(dataset_name)
    client_loaders = partition_data(trainset, num_clients)
    test_loader = get_test_loader(testset)

    input_channels = 1 if dataset_name == 'MNIST' else 3
    global_model = SimpleCNN(input_channels=input_channels).to(device)

    if not os.path.exists('./log'):
        os.makedirs('./log')

    attack_results = []

    for epsilon in epsilon_values:
        dp_accuracies = []
        global_model_dp = SimpleCNN(input_channels=input_channels).to(device)

        for round in range(num_rounds):
            client_models = [train_client(loader, global_model_dp, local_epochs, delta, epsilon)[0] for loader in client_loaders]
            global_model_dp = aggregate_models(global_model_dp, client_models)
            accuracy, _ = test_model(global_model_dp, test_loader)
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

        # Run MIA for each epsilon value
        _, member_logits = train_client(client_loaders[0], global_model_dp, epochs=1, delta=0.002, epsilon=epsilon, record_logits=True)
        _, non_member_logits = test_model(global_model_dp, test_loader, record_logits=True)
        X, y = prepare_attack_data(member_logits, non_member_logits)
        print(f"Running attack model for epsilon {epsilon}...")
        _, attack_accuracy = train_attack_model(X, y)
        attack_results.append((epsilon, attack_accuracy))

    # Create a table for Attack Model Accuracy vs Epsilon Values
    attack_df = pd.DataFrame(attack_results, columns=['Epsilon', 'Attack Model Accuracy'])
    plt.figure(figsize=(8, 4))
    sns.heatmap(attack_df.set_index('Epsilon').T, annot=True, cmap='Blues', cbar=False, fmt='.4f')
    plt.title('Attack Model Accuracy vs Epsilon Values')
    plt.savefig('./log/attack_model_accuracy_table.png')
    plt.close()

    print("All simulations completed and results saved.")

if __name__ == "__main__":
    dataset_name = 'MNIST'
    run_fedavg(dataset_name)
