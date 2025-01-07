import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class Node:
    def __init__(self, node_id, x_data, y_data):
        self.node_id = node_id
        self.x_data = x_data
        self.y_data = y_data
        print(f"\nInitializing Node {node_id} with {len(x_data)} training samples")
        print(f"Label distribution: {np.bincount(y_data.flatten())}")
        self.model = self.create_model()

    def train_local(self, epochs=1):
        print(f"\nNode {self.node_id} starting local training...")
        history = self.model.fit(
            self.x_data, self.y_data,
            epochs=epochs,
            batch_size=32,
            verbose=0
        )
        print(f"Node {self.node_id} completed local training. "
              f"Loss: {history.history['loss'][-1]:.4f}, "
              f"Accuracy: {history.history['accuracy'][-1]:.4f}")

    def evaluate(self, x_test, y_test):
        loss, acc = self.model.evaluate(x_test, y_test, verbose=0)
        print(f"Node {self.node_id} evaluation - Loss: {loss:.4f}, Accuracy: {acc:.4f}")
        return loss, acc

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    @staticmethod
    def create_model():
        inputs = tf.keras.Input(shape=(28, 28, 1))
        x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(10, activation='softmax')(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model


def load_data():
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_test = np.expand_dims(x_test, axis=-1)
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
        x_node = np.expand_dims(x_sorted[start_idx:end_idx], axis=-1)
        y_node = y_sorted[start_idx:end_idx]
        print(f"Node {i} data shape: {x_node.shape}")
        node_data.append((x_node, y_node))

    return node_data


def average_weights(weights1, weights2):
    return [(w1 + w2) / 2.0 for w1, w2 in zip(weights1, weights2)]
