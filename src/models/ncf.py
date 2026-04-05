# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import tensorflow.keras as keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.models.base_model import BaseModel

__all__ = ["NCFModel"]


class NCFModel(BaseModel):
    """Neural Collaborative Filtering (NCF)

    X. He, L. Liao, H. Zhang, L. Nie, X. Hu, and T.-S. Chua,
    "Neural Collaborative Filtering," in the 26th International World
    Wide Web Conference, 2017.

    Attributes:
        hparams (object): Global hyper-parameters.
    """

    def __init__(self, hparams, seed=None):
        """Initialization steps for NCF.
        Calls the BaseModel's __init__ method after initializing required components.

        Args:
            hparams (object): Global hyper-parameters.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
        """
        super().__init__(hparams, seed=seed)

    def _build_graph(self):
        """Build the NCF model.

        Returns:
            object: A model used to train and evaluate NCF.
        """
        model = self._build_ncf()
        scorer= self._build_ncf()
        return model, scorer

    def _build_ncf(self):
        """
        The main function to create the NCF model logic.

        Returns:
            object: A Keras model for Neural Collaborative Filtering.
        """
        hparams = self.hparams

        # Inputs for user and item IDs
        user_input = keras.Input(shape=(1,), name="user_input", dtype="int32")
        item_input = keras.Input(shape=(1,), name="item_input", dtype="int32")

        # Embedding layers for users and items
        user_embedding = layers.Embedding(
            hparams.num_users, hparams.embedding_dim, name="user_embedding"
        )(user_input)  # User embedding size given by hparams.embedding_dim
        item_embedding = layers.Embedding(
            hparams.num_items, hparams.embedding_dim, name="item_embedding"
        )(item_input)  # Item embedding size given by hparams.embedding_dim

        # Flatten embeddings
        user_embedding_flat = layers.Flatten()(user_embedding)
        item_embedding_flat = layers.Flatten()(item_embedding)

        # Generalized Matrix Factorization (GMF)
        gmf_vector = layers.Multiply(name="gmf_layer")([user_embedding_flat, item_embedding_flat])

        # Multi-Layer Perceptron (MLP)
        mlp_vector = layers.Concatenate(name="mlp_input")([user_embedding_flat, item_embedding_flat])
        hidden_layer_sizes = hparams.hidden_layer_sizes  # List of hidden layer sizes
        for i, layer_size in enumerate(hidden_layer_sizes):
            mlp_vector = layers.Dense(
                layer_size, activation="relu", name=f"mlp_layer_{i}"
            )(mlp_vector)

        # Combine GMF and MLP
        combined_vector = layers.Concatenate(name="combined")([gmf_vector, mlp_vector])

        # Output layer for interaction probability
        output = layers.Dense(1, activation="sigmoid", name="output")(combined_vector)

        # Define the NCF model
        model = keras.Model(inputs=[user_input, item_input], outputs=output, name="NCF")
        return model

    def fit(self, input_data, labels, filepath, epochs=1, batch_size=32, pre_ep=0):
        """
        Optimized training function using tf.data for efficient data handling.
        Includes epoch-wise loss tracking and model checkpointing.

        Args:
            input_data: List of two arrays [users, items] representing input features.
            labels: Array of labels corresponding to input data.
            filepath: Path to save model checkpoints.
            epochs: Number of epochs to train the model.
            batch_size: Batch size for training.
        """
        import gc
        import tensorflow as tf
        import numpy as np
        from tqdm import tqdm
        import matplotlib.pyplot as plt

        # Ensure inputs are NumPy arrays
        users, items = input_data
        users = np.array(users)
        items = np.array(items)
        labels = np.array(labels)

        # Create TensorFlow dataset
        train_dataset = tf.data.Dataset.from_tensor_slices(((users, items), labels))
        train_dataset = (
            train_dataset.shuffle(buffer_size=10000)  # Shuffle to ensure randomness
            .batch(batch_size)  # Batch size
            .prefetch(tf.data.AUTOTUNE)  # Prefetch for efficient GPU utilization
        )

        # Initialize training loss tracker
        train_losses = []

        # Training loop
        for epoch in tqdm(range(epochs), desc="Epochs", unit="epoch"):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # Initialize batch-wise loss tracker
            epoch_train_loss = []

            for (X_batch, y_batch) in tqdm(train_dataset, desc="Training", unit="batch", leave=False):
                # Train on batch
                loss = self.model.train_on_batch(X_batch, y_batch)
                epoch_train_loss.append(loss)

            # Compute and log epoch loss
            train_loss = np.mean(epoch_train_loss)
            train_losses.append(train_loss)
            print(f"Epoch {epoch + 1} - Loss: {train_loss:.4f}")

            # Save model weights at the end of the epoch
            self.model.save_weights(f"{filepath}model/ep_{epoch + 1 + pre_ep}.weights.h5")

            # Clear memory after the epoch
            gc.collect()
            tf.keras.backend.clear_session()

        # Plot and save training loss curve
        plt.figure(figsize=(12, 5))
        plt.plot(train_losses, marker='o', linestyle='-', label="Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{filepath}/loss_curve.png")
        plt.show()



    def predict(self, input_data, batch_size=32):
        """
        Predict interaction probabilities for user-item pairs.

        Args:
            input_data: Tuple of two arrays (users, items) representing input features.
            batch_size: Batch size for predictions.

        Returns:
            numpy.ndarray: Predicted interaction probabilities for the input pairs.
        """
        # Ensure inputs are NumPy arrays
        users, items = input_data
        users = np.array(users)
        items = np.array(items)

        # Calculate the number of batches
        num_batches = len(users) // batch_size

        # Initialize a list to store predictions
        predictions = []

        # Use tqdm to display progress for batches
        for batch_index in tqdm(range(num_batches), desc="Batches", unit="batch"):
            # Create each batch
            batch_users = users[batch_index * batch_size:(batch_index + 1) * batch_size]
            batch_items = items[batch_index * batch_size:(batch_index + 1) * batch_size]

            # Predict on batch
            batch_preds = self.model.predict_on_batch([batch_users, batch_items])
            predictions.append(batch_preds)

        # Handle the last batch if the total data is not perfectly divisible by batch_size
        if len(users) % batch_size != 0:
            batch_users = users[num_batches * batch_size:]
            batch_items = items[num_batches * batch_size:]
            if len(batch_users) > 0:
                batch_preds = self.model.predict_on_batch([batch_users, batch_items])
                predictions.append(batch_preds)

        # Concatenate all batch predictions into a single array
        predictions = np.concatenate(predictions, axis=0)

        return predictions


if __name__ == "__main__":
    # Example hyperparameters
    class HParams:
        num_users = 5000
        num_items = 50000
        embedding_dim = 64
        num_hidden_layers = 3
        hidden_layer_size = 128


    hparams = HParams()

    # Create and build the NCF model
    ncf = NCFModel(hparams)
    model = ncf._build_graph()

    # Compile the model
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Summary of the model
    model.summary()
