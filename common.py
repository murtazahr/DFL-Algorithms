import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(576, 64)  # 576 = 64 * 3 * 3
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Node:
    def __init__(self, node_id, x_data, y_data):
        self.node_id = node_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Convert numpy arrays to PyTorch tensors
        self.x_data = torch.FloatTensor(x_data).to(self.device)
        self.y_data = torch.LongTensor(y_data).to(self.device)

        print(f"\nInitializing Node {node_id} with {len(x_data)} training samples")
        print(f"Label distribution: {np.bincount(y_data.flatten())}")

        self.model = ConvNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss()

    def train_local(self, epochs=1):
        print(f"\nNode {self.node_id} starting local training...")
        self.model.train()

        dataset = torch.utils.data.TensorDataset(self.x_data, self.y_data)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        total_loss = 0
        correct = 0
        total = 0

        for epoch in range(epochs):
            for batch_x, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()

        accuracy = correct / total
        avg_loss = total_loss / len(dataloader)
        print(f"Node {self.node_id} completed local training. "
              f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    def evaluate(self, x_test, y_test):
        self.model.eval()
        x_test = torch.FloatTensor(x_test).to(self.device)
        y_test = torch.LongTensor(y_test).to(self.device)

        with torch.no_grad():
            outputs = self.model(x_test)
            loss = self.criterion(outputs, y_test)
            _, predicted = outputs.max(1)
            correct = predicted.eq(y_test).sum().item()
            accuracy = correct / len(y_test)

        print(f"Node {self.node_id} evaluation - Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")
        return loss.item(), accuracy

    def get_weights(self):
        return {name: param.data.clone() for name, param in self.model.named_parameters()}

    def set_weights(self, weights):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.data.copy_(weights[name])


def load_data():
    print("Loading MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST('./data', train=False, download=True, transform=transform)

    x_train = mnist_train.data.numpy()[:, None, :, :] / 255.0  # Add channel dimension
    y_train = mnist_train.targets.numpy()
    x_test = mnist_test.data.numpy()[:, None, :, :] / 255.0
    y_test = mnist_test.targets.numpy()

    return (x_train, y_train), (x_test, y_test)


def split_data(x_train, y_train, num_nodes):
    print(f"\nSplitting data among {num_nodes} nodes...")
    sorted_idx = np.argsort(y_train)
    x_sorted = x_train[sorted_idx]
    y_sorted = y_train[sorted_idx]

    samples_per_node = len(x_train) // num_nodes
    node_data = []

    for i in range(num_nodes):
        start_idx = i * samples_per_node
        end_idx = (i + 1) * samples_per_node
        x_node = x_sorted[start_idx:end_idx]
        y_node = y_sorted[start_idx:end_idx]
        print(f"Node {i} data shape: {x_node.shape}")
        node_data.append((x_node, y_node))

    return node_data


def average_weights(weights1, weights2):
    return {name: (weights1[name] + weights2[name]) / 2.0
            for name in weights1.keys()}
