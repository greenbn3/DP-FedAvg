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
    for epoch in range(epochs):
        for data, target in loader:
            data, target = data.cpu(), target.cpu()
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            
            if record_logits:
                train_logits.append((output.detach().cpu().numpy(), target.cpu().numpy()))  # Save logits for later
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
            data, target = data.cpu(), target.cpu()
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            if record_logits:
                test_logits.append((output.cpu().numpy(), target.cpu().numpy()))  # Save logits
    return 100. * correct / len(test_loader.dataset), test_logits if record_logits else 100. * correct / len(test_loader.dataset)

def train_shadow_models(trainset, num_shadow_models=5, num_clients=4):
    shadow_models = []
    shadow_client_loaders = [partition_data(trainset, num_clients=num_clients) for _ in range(num_shadow_models)]
    
    for shadow_loader in shadow_client_loaders:
        shadow_model = SimpleCNN().cpu()
        client_models = [train_client(loader, shadow_model, epochs=1, delta=0.002, epsilon=10.0)[0] for loader in shadow_loader]
        shadow_model = aggregate_models(shadow_model, client_models)
        shadow_models.append(shadow_model)
    
    return shadow_models

def prepare_attack_data(member_logits, non_member_logits):
    X = np.concatenate([member_logits, non_member_logits])
    y = np.concatenate([np.ones(len(member_logits)), np.zeros(len(non_member_logits))])
    return X, y

def train_attack_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    attack_model = LogisticRegression()
    attack_model.fit(X_train, y_train)
    predictions = attack_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Attack Model Accuracy: {accuracy:.4f}")
    return attack_model

def run_fedavg(dataset_name='CIFAR10', num_clients=4, num_rounds=150, local_epochs=1, epsilon_values=[0.0, 1.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0], delta=0.002):
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
            client_models = [train_client(loader, global_model_dp, local_epochs, delta, epsilon)[0] for loader in client_loaders]
            global_model_dp = aggregate_models(global_model_dp, client_models)
            accuracy, _ = test_model(global_model_dp, test_loader)  # Extract accuracy, ignore logits
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

    # Run MIA
    trainset, testset = load_dataset('CIFAR10')
    client_loaders = partition_data(trainset, num_clients=4)
    test_loader = get_test_loader(testset)
    global_model_dp = SimpleCNN().cpu()

    # Collect member logits
    _, member_logits = train_client(client_loaders[0], global_model_dp, epochs=1, delta=0.002, epsilon=10.0, record_logits=True)

    # Collect non-member logits
    _, non_member_logits = test_model(global_model_dp, test_loader, record_logits=True)

    # Prepare and run attack model
    X, y = prepare_attack_data(member_logits, non_member_logits)
    attack_model = train_attack_model(X, y)

