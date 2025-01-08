import random
import time
from datetime import datetime

import numpy as np

from common import Node, load_data, split_data


class OpportunisticNode(Node):
    def __init__(self, node_id, x_data, y_data, similarity_threshold=0.5):
        super().__init__(node_id, x_data, y_data)
        self.similarity_threshold = similarity_threshold
        self.label_distribution = self.calculate_label_distribution()

    def calculate_label_distribution(self):
        """Calculate normalized distribution of labels in local data"""
        y_numpy = self.y_data.cpu().numpy()
        # Ensure we count all possible labels (0-9 for MNIST)
        dist = np.bincount(y_numpy.flatten(), minlength=10)
        # Add small constant to avoid division by zero
        dist = dist + 1e-10
        return dist / np.sum(dist)

    def calculate_similarity(self, peer_distribution):
        """Calculate cosine similarity between label distributions"""
        similarity = np.dot(self.label_distribution, peer_distribution) / (
                np.linalg.norm(self.label_distribution) * np.linalg.norm(peer_distribution)
        )
        print(f"Detailed similarity calculation:")
        print(f"Self distribution: {self.label_distribution}")
        print(f"Peer distribution: {peer_distribution}")
        print(f"Computed similarity: {similarity}")
        return similarity

    def evaluate_peer_utility(self, peer_weights, x_val, y_val):
        """Evaluate how well peer's model performs on validation data"""
        original_weights = self.get_weights()  # Save current weights
        self.set_weights(peer_weights)  # Try peer's weights
        loss, acc = self.evaluate(x_val, y_val)
        self.set_weights(original_weights)  # Restore original weights
        return acc

    def selective_update(self, peer_weights, similarity_score):
        """Selectively update model based on similarity"""
        print(f"Node {self.node_id} considering update. "
              f"Similarity: {similarity_score:.4f}, "
              f"Threshold: {self.similarity_threshold:.4f}")

        if similarity_score > self.similarity_threshold:
            print(f"Update approved for node {self.node_id}")
            # Weight the update based on similarity
            current_weights = self.get_weights()
            weighted_weights = {}
            for name in current_weights.keys():
                # More similar peers have more influence
                weighted_weights[name] = (similarity_score * peer_weights[name] +
                                          (1 - similarity_score) * current_weights[name])
            self.set_weights(weighted_weights)
            return True
        print(f"Update rejected for node {self.node_id}")
        return False


def run_opportunistic_learning(nodes, num_rounds, x_test, y_test, peers_per_round=2):
    num_nodes = len(nodes)
    hist = []
    start_time = time.time()

    # Split test data for validation
    val_size = len(x_test) // 5
    x_val, y_val = x_test[:val_size], y_test[:val_size]
    x_test, y_test = x_test[val_size:], y_test[val_size:]

    print(f"\n{'=' * 50}")
    print(f"Starting Opportunistic Learning with {num_nodes} nodes")
    print(f"Similarity threshold: {nodes[0].similarity_threshold}")
    print(f"{'=' * 50}")

    for round_idx in range(num_rounds):
        round_start = time.time()
        print(f"\nRound {round_idx + 1}/{num_rounds}")
        print("-" * 30)

        updates_made = 0

        # Each node considers some peers
        for i in range(num_nodes):
            # Select random peers to evaluate
            peers = random.sample([j for j in range(num_nodes) if j != i],
                                  peers_per_round)

            print(f"\nNode {i} evaluating peers {peers}")

            for peer_idx in peers:
                # Calculate data distribution similarity
                peer_dist = nodes[peer_idx].label_distribution
                similarity = nodes[i].calculate_similarity(peer_dist)

                print(f"Similarity with peer {peer_idx}: {similarity:.4f}")

                # If similar enough, evaluate model utility
                if similarity > nodes[i].similarity_threshold:
                    peer_weights = nodes[peer_idx].get_weights()
                    utility = nodes[i].evaluate_peer_utility(peer_weights, x_val, y_val)
                    print(f"Utility of peer {peer_idx}: {utility:.4f}")

                    # Try to update based on similarity
                    if nodes[i].selective_update(peer_weights, similarity):
                        updates_made += 1
                        # Local training after update
                        nodes[i].train_local()
                        print(f"Node {i} updated using peer {peer_idx}")

        print(f"\nUpdates made this round: {updates_made}")

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
            'min_acc': min(accuracies),
            'updates_made': updates_made
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
    similarity_threshold = 0.01  # Minimum similarity to consider update
    peers_per_round = 2  # Number of peers each node evaluates

    # Split data among nodes
    node_data = split_data(x_train, y_train, num_nodes)

    # Create opportunistic nodes
    print("\nCreating nodes...")
    nodes = [OpportunisticNode(i, data[0], data[1], similarity_threshold)
             for i, data in enumerate(node_data)]

    # Run opportunistic learning
    hist = run_opportunistic_learning(nodes, num_rounds, x_test, y_test,
                                      peers_per_round)

    return hist


if __name__ == "__main__":
    print(f"\nStarting DFL simulation at {datetime.now()}")
    history = main()
    print(f"\nSimulation completed at {datetime.now()}")
