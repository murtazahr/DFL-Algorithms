import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
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


class DPNode:
    def __init__(self, node_id, x_data, y_data, epsilon=1.0, delta=1e-5):
        self.node_id = node_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epsilon = epsilon
        self.delta = delta

        # Convert numpy arrays to PyTorch tensors
        self.x_data = torch.FloatTensor(x_data).to(self.device)
        self.y_data = torch.LongTensor(y_data).to(self.device)

        print(f"\nInitializing DP Node {node_id} with {len(x_data)} training samples")
        print(f"Label distribution: {np.bincount(y_data.flatten(), minlength=3)}")

        # Initialize model and make it compatible with DP
        self.model = SoilNet().to(self.device)
        self.model = ModuleValidator.fix(self.model)  # Make model DP-compatible

        # Set up optimizer
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss()

        # Create DataLoader first
        self.dataset = torch.utils.data.TensorDataset(self.x_data, self.y_data)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=32,
            shuffle=True
        )

        # Initialize privacy engine
        self.privacy_engine = PrivacyEngine(secure_mode=False)  # Development mode

        # Sample size affects privacy analysis
        sample_rate = self.dataloader.batch_size / len(self.dataset)

        # Attach privacy engine to model, optimizer, and dataloader
        self.model, self.optimizer, self.dataloader = self.privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.dataloader,
            target_epsilon=self.epsilon,
            target_delta=self.delta,
            epochs=50,
            max_grad_norm=1.0,
            poisson_sampling=False,  # Improves privacy guarantees
            batch_first=True
        )

    def train_local(self, epochs=50):
        print(f"\nNode {self.node_id} starting local DP training...")
        self.model.train()

        total_loss = 0
        correct = 0
        total = 0

        for epoch in range(epochs):
            for batch_x, batch_y in self.dataloader:
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
        avg_loss = total_loss / len(self.dataloader)

        # Get privacy spent
        epsilon = self.privacy_engine.accountant.get_epsilon(delta=self.delta)

        print(f"\nNode {self.node_id} completed local training:")
        print(f"Performance metrics:")
        print(f"- Loss: {avg_loss:.4f}")
        print(f"- Accuracy: {accuracy:.4f}")
        print(f"Privacy guarantee:")
        print(f"- (ε, δ)-DP with ε = {epsilon:.2f}, δ = {self.delta:.2e}")
        if hasattr(self.optimizer, 'noise_multiplier'):
            print(f"- Noise multiplier: {self.optimizer.noise_multiplier:.2f}")
        print(f"- Gradient clipping norm: {self.optimizer.max_grad_norm:.2f}")

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
    """Load and preprocess the soil dataset"""
    print("Loading soil dataset...")
    df = pd.read_csv('OrgSoil.csv')

    X = df.drop(['Output'], axis=1).values
    y = df['Output'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    return (X_train, y_train), (X_test, y_test)


def split_data(x_train, y_train, num_nodes):
    """Split data among nodes"""
    print(f"\nSplitting data among {num_nodes} nodes...")
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
    """Average two sets of weights"""
    if not weights1 or not weights2:
        raise ValueError("Empty weights received for averaging")

    if set(weights1.keys()) != set(weights2.keys()):
        raise ValueError("Weight dictionaries have different keys")

    averaged = {}
    for name in weights1.keys():
        if weights1[name].shape != weights2[name].shape:
            raise ValueError(f"Weight shapes don't match for {name}: {weights1[name].shape} vs {weights2[name].shape}")
        # Ensure both tensors are float32 before averaging
        w1 = weights1[name].to(torch.float32)
        w2 = weights2[name].to(torch.float32)
        averaged[name] = (w1 + w2) / 2.0

    return averaged
