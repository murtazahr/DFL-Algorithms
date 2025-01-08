import random
import time
from datetime import datetime

import numpy as np

from soil_federated_ldp import DPNode, load_data, split_data, average_weights


def run_pairwise_gossip(nodes, num_rounds, x_test, y_test):
    num_nodes = len(nodes)
    hist = []
    start_time = time.time()

    print(f"\n{'=' * 50}")
    print(f"Starting Pairwise Gossip Learning with {num_nodes} nodes")
    print(f"{'=' * 50}")

    for round_idx in range(num_rounds):
        round_start = time.time()
        print(f"\nRound {round_idx + 1}/{num_rounds}")
        print("-" * 30)

        # Keep track of which nodes have been updated this round
        updated_nodes = set()

        # Random peer selection and model averaging
        for i in range(num_nodes):
            if i in updated_nodes:
                continue

            # Select a peer that hasn't been updated yet
            available_peers = [j for j in range(num_nodes)
                               if j != i and j not in updated_nodes]
            if not available_peers:
                continue

            peer_idx = random.choice(available_peers)
            print(f"\nNode {i} paired with Node {peer_idx}")

            try:
                # Get weights from both nodes
                weights_i = nodes[i].get_weights()
                weights_j = nodes[peer_idx].get_weights()

                # Verify weights are valid
                if not weights_i or not weights_j:
                    print(f"Warning: Empty weights detected for nodes {i} and {peer_idx}")
                    continue

                print(f"Node {i} weights: {len(weights_i)} layers")
                print(f"Node {peer_idx} weights: {len(weights_j)} layers")

                # Average the weights
                averaged_weights = average_weights(weights_i, weights_j)

                # Update both nodes
                nodes[i].set_weights(averaged_weights)
                nodes[peer_idx].set_weights(averaged_weights)

                # Local training
                nodes[i].train_local()
                nodes[peer_idx].train_local()

                # Mark both nodes as updated
                updated_nodes.add(i)
                updated_nodes.add(peer_idx)

            except Exception as e:
                print(f"Error during weight averaging between nodes {i} and {peer_idx}: {str(e)}")
                continue

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

    # Split data among nodes
    node_data = split_data(x_train, y_train, num_nodes)

    # Create nodes
    print("\nCreating nodes...")
    nodes = [DPNode(i, data[0], data[1], epsilon=10.0, delta=0.001)
             for i, data in enumerate(node_data)]

    # Run pairwise gossip
    hist = run_pairwise_gossip(nodes, num_rounds, x_test, y_test)

    return hist


if __name__ == "__main__":
    print(f"\nStarting DFL simulation at {datetime.now()}")
    history = main()
    print(f"\nSimulation completed at {datetime.now()}")
