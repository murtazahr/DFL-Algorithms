import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class SoilNet(nn.Module):
    def __init__(self):
        super(SoilNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )

    def forward(self, x):
        return self.layers(x)


class Node:
    def __init__(self, node_id, x_data, y_data):
        self.node_id = node_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Convert numpy arrays to PyTorch tensors
        self.x_data = torch.FloatTensor(x_data).to(self.device)
        self.y_data = torch.LongTensor(y_data).to(self.device)

        print(f"\nInitializing Node {node_id} with {len(x_data)} training samples")
        print(f"Label distribution: {np.bincount(y_data.flatten(), minlength=3)}")

        self.model = SoilNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss()

    def train_local(self, epochs=50):
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
    print("Loading soil dataset...")
    # Read the CSV file
    df = pd.read_csv('OrgSoil.csv')

    # Separate features and target
    X = df.drop(['Output'], axis=1).values
    y = df['Output'].values

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    return (X_train, y_train), (X_test, y_test)


def split_data(x_train, y_train, num_nodes):
    print(f"\nSplitting data among {num_nodes} nodes...")
    # Shuffle data before splitting to ensure random distribution
    indices = np.random.permutation(len(x_train))
    x_shuffled = x_train[indices]
    y_shuffled = y_train[indices]

    samples_per_node = len(x_train) // num_nodes
    node_data = []

    for i in range(num_nodes):
        start_idx = i * samples_per_node
        end_idx = (i + 1) * samples_per_node
        x_node = x_shuffled[start_idx:end_idx]
        y_node = y_shuffled[start_idx:end_idx]
        print(f"Node {i} data shape: {x_node.shape}")
        node_data.append((x_node, y_node))

    return node_data


def average_weights(weights1, weights2):
    """Average two sets of weights with validation and error checking"""
    if not weights1 or not weights2:
        raise ValueError("Empty weights received for averaging")

    if set(weights1.keys()) != set(weights2.keys()):
        raise ValueError("Weight dictionaries have different keys")

    averaged = {}
    for name in weights1.keys():
        if weights1[name].shape != weights2[name].shape:
            raise ValueError(f"Weight shapes don't match for {name}: {weights1[name].shape} vs {weights2[name].shape}")
        averaged[name] = (weights1[name] + weights2[name]) / 2.0

    if not averaged:
        raise ValueError("No weights were averaged")

    return averaged
