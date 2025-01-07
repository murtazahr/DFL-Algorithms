import random
import time
from datetime import datetime
import numpy as np
from common import Node, load_data, split_data


class BroadcastNode(Node):
    def __init__(self, node_id, x_data, y_data, buffer_size=5):
        super().__init__(node_id, x_data, y_data)
        self.buffer_size = buffer_size
        self.weight_buffer = []  # Store received model weights

    def add_to_buffer(self, weights):
        """Add weights to buffer, return True if buffer is full"""
        if len(self.weight_buffer) < self.buffer_size:
            self.weight_buffer.append(weights)
            print(f"Node {self.node_id} buffer: {len(self.weight_buffer)}/{self.buffer_size}")
            return len(self.weight_buffer) == self.buffer_size
        return True

    def aggregate_and_update(self):
        """Average all weights in buffer and update model"""
        if not self.weight_buffer:
            return

        print(f"Node {self.node_id} aggregating {len(self.weight_buffer)} models")
        # Average weights across all dimensions
        avg_weights = []
        for weights_idx in range(len(self.weight_buffer[0])):
            layer_weights = [weights[weights_idx] for weights in self.weight_buffer]
            avg_weights.append(np.mean(layer_weights, axis=0))

        # Update model and clear buffer
        self.set_weights(avg_weights)
        self.weight_buffer = []
        print(f"Node {self.node_id} cleared buffer after aggregation")


def run_broadcast_gossip(nodes, num_rounds, x_test, y_test, broadcasts_per_round=2):
    num_nodes = len(nodes)
    hist = []
    start_time = time.time()

    print(f"\n{'=' * 50}")
    print(f"Starting Broadcast Gossip Learning with {num_nodes} nodes")
    print(f"Buffer size: {nodes[0].buffer_size}, Broadcasts per round: {broadcasts_per_round}")
    print(f"{'=' * 50}")

    for round_idx in range(num_rounds):
        round_start = time.time()
        print(f"\nRound {round_idx + 1}/{num_rounds}")
        print("-" * 30)

        # Each node broadcasts to some peers
        for i in range(num_nodes):
            # Select random peers to broadcast to
            num_peers = min(broadcasts_per_round, num_nodes - 1)
            peers = random.sample([j for j in range(num_nodes) if j != i], num_peers)

            # Broadcast current weights to selected peers
            current_weights = nodes[i].get_weights()
            print(f"\nNode {i} broadcasting to nodes {peers}")

            for peer_idx in peers:
                # Add weights to peer's buffer
                buffer_full = nodes[peer_idx].add_to_buffer(current_weights)

                # If buffer is full, aggregate and train
                if buffer_full:
                    nodes[peer_idx].aggregate_and_update()
                    nodes[peer_idx].train_local()

        # Force remaining nodes to aggregate if they have anything in buffer
        for node in nodes:
            if node.weight_buffer:
                node.aggregate_and_update()
                node.train_local()

        # Evaluate all nodes
        print("\nEvaluating all nodes...")
        accuracies = []
        for node in nodes:
            _, acc = node.evaluate(x_test, y_test)
            accuracies.append(acc)

        avg_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)

        print(f"\nRound {round_idx + 1} Summary:")
        print(f"Average Accuracy: {avg_acc:.4f} ± {std_acc:.4f}")
        print(f"Best Node Accuracy: {max(accuracies):.4f}")
        print(f"Worst Node Accuracy: {min(accuracies):.4f}")
        print(f"Round took {time.time() - round_start:.2f} seconds")

        hist.append({
            'round': round_idx + 1,
            'avg_acc': avg_acc,
            'std_acc': std_acc,
            'max_acc': max(accuracies),
            'min_acc': min(accuracies)
        })

    total_time = time.time() - start_time
    print(f"\n{'=' * 50}")
    print(f"Training completed in {total_time:.2f} seconds")
    print(f"Final Average Accuracy: {hist[-1]['avg_acc']:.4f}")
    print(f"{'=' * 50}")

    return hist


def main():
    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = load_data()

    # Parameters
    num_nodes = 5
    num_rounds = 10
    buffer_size = 3  # Number of models to collect before averaging
    broadcasts_per_round = 2  # Number of peers each node broadcasts to

    # Split data among nodes
    node_data = split_data(x_train, y_train, num_nodes)

    # Create broadcast nodes
    print("\nCreating nodes...")
    nodes = [BroadcastNode(i, data[0], data[1], buffer_size)
             for i, data in enumerate(node_data)]

    # Run broadcast gossip
    hist = run_broadcast_gossip(nodes, num_rounds, x_test, y_test,
                                broadcasts_per_round)

    return hist


if __name__ == "__main__":
    print(f"\nStarting DFL simulation at {datetime.now()}")
    history = main()
    print(f"\nSimulation completed at {datetime.now()}")
