

import time
import abc
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras  # Use tensorflow.keras instead of compat.v1.keras
import gc

from sklearn.metrics import mean_squared_error
from src.evaluation.recomendation_eval import cal_metric       ##???


__all__ = ["BaseModel"]


class BaseModel:
    """Basic class of models

    Attributes:
        hparams (HParams): A HParams object, holds the entire set of hyperparameters.
        train_iterator (object): An iterator to load the data in training steps.
        test_iterator (object): An iterator to load the data in testing steps.
        graph (object): An optional graph.
        seed (int): Random seed.
    """

    def __init__(
        self,
        hparams,
        seed=None,
    ):
        """Initializing the models. Create common logics which are needed by all deeprec models, such as loss function,
        parameter set.

        Args:
            hparams (HParams): A HParams object, holds the entire set of hyperparameters.
            iterator_creator (object): An iterator to load the data.
            graph (object): An optional graph.
            seed (int): Random seed.
        """
        self.seed = seed
        # tf.compat.v1.set_random_seed(seed)
        tf.random.set_seed(seed)

        np.random.seed(seed)

        self.hparams = hparams
        self.support_quick_scoring = hparams.support_quick_scoring



        # Set GPU memory growth (only needed if you want dynamic memory allocation)
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Set memory growth to allow TensorFlow to allocate memory as needed
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)

        # No need to set a manual session; TensorFlow 2.x manages this automatically

        # Your model-building function (this should be defined elsewhere)
        self.model, self.scorer = self._build_graph()

        self.loss = self._get_loss()
        self.train_optimizer = self._get_opt()

        self.model.compile(loss=self.loss, optimizer=self.train_optimizer)

    def _init_embedding(self, file_path):
        """Load pre-trained embeddings as a constant tensor.

        Args:
            file_path (str): the pre-trained glove embeddings file path.

        Returns:
            numpy.ndarray: A constant numpy array.
        """

        return np.load(file_path)

    @abc.abstractmethod
    def _build_graph(self):
        """Subclass will implement this."""
        pass

    @abc.abstractmethod
    def _get_input_label_from_iter(self, batch_data):
        """Subclass will implement this"""
        pass

    def _get_loss(self):
        """Make loss function, consists of data loss and regularization loss

        Returns:
            object: Loss function or loss function name
        """
        if self.hparams.loss == "cross_entropy_loss":
            data_loss = "categorical_crossentropy"
        elif self.hparams.loss == "log_loss":
            data_loss = "binary_crossentropy"
        else:
            raise ValueError("this loss not defined {0}".format(self.hparams.loss))
        return data_loss

    def _get_opt(self):
        """Get the optimizer according to configuration. Usually we will use Adam.
        Returns:
            object: An optimizer.
        """
        lr = self.hparams.learning_rate
        optimizer = self.hparams.optimizer

        if optimizer == "adam":
            train_opt = keras.optimizers.Adam(learning_rate=lr)

        return train_opt

    def train_val_split(self,input_data, labels, val_split=0.2):
        """
        Splits the data into training and validation sets.

        Parameters:
        - input_data: A list of two numpy arrays, each with the same number of samples (e.g., [X1, X2])
        - labels: The labels corresponding to the input data
        - val_split: The fraction of data to be used as validation data (default is 0.2)

        Returns:
        - X_train: List of training data arrays
        - X_val: List of validation data arrays
        - y_train: Training labels
        - y_val: Validation labels
        """
        # Determine the number of samples (assuming both arrays have the same size)
        num_samples = len(labels)

        # Calculate the number of validation samples
        num_val_samples = int(val_split * num_samples)

        # Shuffle the data
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        # Split the indices for training and validation
        val_indices = indices[:num_val_samples]
        train_indices = indices[num_val_samples:]

        # Split the input data and labels into training and validation sets
        X_train = [data[train_indices] for data in input_data]
        X_val = [data[val_indices] for data in input_data]
        y_train = labels[train_indices]
        y_val = labels[val_indices]

        return X_train, X_val, y_train, y_val




    def fit(self, input_data, labels, filepath, epochs=1, batch_size=32):
        """
    Optimized training function to avoid symbolic tensor issues with train_on_batch.
    """
        """
                    Optimized training function for large datasets using tf.data.
                    Saves model weights at the end of every epoch and ensures efficient GPU utilization.

                    Args:
                        input_data: List of two arrays [history, candidate] representing input features.
                        labels: Array of labels corresponding to input data.
                        filepath: Path to save model checkpoints.
                        epochs: Number of epochs to train the model.
                        batch_size: Batch size for training.
                    """
        # Unpack inputs
        history, candidate = input_data

        # Create TensorFlow dataset for efficient data handling
        train_dataset = tf.data.Dataset.from_tensor_slices(((history, candidate), labels))
        train_dataset = (
            train_dataset.shuffle(buffer_size=1000)  # Reduce shuffle buffer size to save memory
            .batch(batch_size)  # Batch size
            .prefetch(1)  # Prefetch only one batch to limit memory usage
        )

        # Initialize variable for tracking training loss
        train_losses = []

        # Training loop
        for epoch in tqdm(range(epochs), desc="Epochs", unit="epoch"):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # Training step
            epoch_train_loss = []
            for (X_batch, y_batch) in tqdm(train_dataset, desc="Training", unit="batch", leave=False):
                # Train on batch
                loss = self.model.train_on_batch(X_batch, y_batch)
                epoch_train_loss.append(loss)

            # Compute average loss for the epoch
            train_loss = np.mean(epoch_train_loss)
            train_losses.append(train_loss)

            print(f"Train Loss: {train_loss:.4f}")

            # Save model weights for the current epoch
            self.model.save_weights(f"{filepath}model/ep_{epoch + 1}.weights.h5")
            self.scorer.save_weights(f"{filepath}scorer/ep_{epoch + 1}.weights.h5")

            # Clear memory after epoch
            gc.collect()
            tf.keras.backend.clear_session()

        # Plot training loss
        plt.figure(figsize=(12, 5))
        plt.plot(train_losses, label="Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{filepath}/loss_curve.png")
        plt.show()




    def predict_new(self, input_data, batch_size=128):
        """
           Optimized predict function with support for batching and handling uneven batch sizes.

           Args:
               input_data (list): A list of two ndarrays [history, candidate].
               batch_size (int): Batch size for prediction.

           Returns:
               np.ndarray: Predicted values concatenated into a single array.
           """
        # Unpack input data
        X_test = input_data  # X_test is a list of two ndarrays [history, candidate]

        # Calculate the number of batches
        num_batches = len(X_test[0]) // batch_size

        # Initialize a list to store predictions
        predictions = []

        # Process batches
        for batch_index in tqdm(range(num_batches), desc="Batches", unit="batch"):
            # Create batch slices for history and candidate inputs
            X_batch_1 = X_test[0][batch_index * batch_size:(batch_index + 1) * batch_size]
            X_batch_2 = X_test[1][batch_index * batch_size:(batch_index + 1) * batch_size]

            # Predict on batch using scorer
            batch_predictions = self.scorer.predict_on_batch([X_batch_1, X_batch_2])

            # Collect predictions
            predictions.append(batch_predictions)

        # Handle the last batch (if any remaining samples)
        if len(X_test[0]) % batch_size != 0:
            X_batch_1 = X_test[0][num_batches * batch_size:]
            X_batch_2 = X_test[1][num_batches * batch_size:]
            if len(X_batch_1) > 0:
                batch_predictions = self.scorer.predict_on_batch([X_batch_1, X_batch_2])
                predictions.append(batch_predictions)

        # Concatenate predictions into a single array
        predictions = np.concatenate(predictions, axis=0)

        return predictions

    def evaluate_rec1(self, y_test, y_pred):
        results= cal_metric(y_test, y_pred, ["auc", 'acc', 'f1', 'precision'])
        print(results)
        return results

    def evaluate_rec2(self, y_test, y_pred): #'group_auc', "hit@5;10", 'ndcg@5;10', 'CTR'
        results= cal_metric(y_test, y_pred, ['CTR'])
        return results


    def user(self, batch_user_input):
        user_input = self._get_user_feature_from_iter(batch_user_input)
        user_vec = self.userencoder.predict_on_batch(user_input)
        user_index = batch_user_input["impr_index_batch"]

        return user_index, user_vec

    def news(self, batch_news_input):
        news_input = self._get_news_feature_from_iter(batch_news_input)
        news_vec = self.newsencoder.predict_on_batch(news_input)
        news_index = batch_news_input["news_index_batch"]

        return news_index, news_vec



    def run_news(self, news_filename):
        if not hasattr(self, "newsencoder"):
            raise ValueError("models must have attribute newsencoder")

        news_indexes = []
        news_vecs = []
        for batch_data_input in tqdm(
            self.test_iterator.load_news_from_file(news_filename)
        ):
            news_index, news_vec = self.news(batch_data_input)
            news_indexes.extend(np.reshape(news_index, -1))
            news_vecs.extend(news_vec)

        return dict(zip(news_indexes, news_vecs))


