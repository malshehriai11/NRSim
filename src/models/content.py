
import time

import tensorflow.keras as keras
from tensorflow.keras import layers
from src.models.base_model import BaseModel

__all__ = ["ContentBasedRecommender"]

class ContentBasedRecommender(BaseModel):
    """Content-Based Recommender with CNN-based News Encoder.

    This model uses word embeddings to represent news articles and a CNN-based news encoder
    for feature extraction.

    Attributes:
        word_embedding (numpy.ndarray): Pretrained word embedding matrix.
        hparams (object): Global hyper-parameters.
    """

    def __init__(self, hparams, seed=None):
        """Initialization for the Content-Based Recommender.

        Args:
            hparams (object): Hyperparameters such as embedding_dim and word_embedding_file.
            seed (int, optional): Random seed.
        """
        self.word_embedding = self._init_embedding(hparams.word_embedding_file)
        super().__init__(hparams, seed=seed)

    def _init_embedding(self, embedding_file):
        """Load the pretrained word embeddings."""
        import numpy as np
        return np.load(embedding_file)

    def _build_news_encoder(self):
        """Create the news encoder with CNN layers.

        Returns:
            keras.Model: The news encoder model.
        """
        hparams = self.hparams

        # Input for the news title
        sequences_input_title = keras.Input(shape=(hparams.title_size,), dtype="int32")

        # Embedding layer
        embedding_layer = layers.Embedding(
            input_dim=self.word_embedding.shape[0],
            output_dim=self.word_embedding.shape[1],
            weights=[self.word_embedding],
            trainable=False,
        )
        embedded_sequences_title = embedding_layer(sequences_input_title)

        # Dropout after embedding
        y = layers.Dropout(hparams.dropout)(embedded_sequences_title)

        # CNN Layer
        y = layers.Conv1D(
            filters=hparams.filter_num,
            kernel_size=hparams.window_size,
            activation=hparams.cnn_activation,
            padding="same",
            bias_initializer=keras.initializers.Zeros(),
            kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
        )(y)

        # Dropout after CNN
        y = layers.Dropout(hparams.dropout)(y)

        # Global pooling to aggregate features
        pred_title = layers.GlobalAveragePooling1D()(y)

        # Build and return the news encoder model
        model = keras.Model(sequences_input_title, pred_title, name="news_encoder")
        return model

    def _build_graph(self):
        """Build the full recommendation pipeline.

        Returns:
            tuple: Training and inference models.
        """
        hparams = self.hparams

        # Build the CNN-based news encoder
        news_encoder = self._build_news_encoder()

        # Input for the article
        article_input = keras.Input(shape=(hparams.title_size,), dtype="int32")
        encoded_article = news_encoder(article_input)

        # Logistic regression for prediction
        prediction = layers.Dense(1, activation="sigmoid")(encoded_article)

        # Define train and scorer models
        train_model = keras.Model(inputs=article_input, outputs=prediction)
        scorer = train_model  # Same model used for scoring

        return train_model, scorer

    # Existing fit_all and predict methods remain unchanged, as they fit seamlessly with the updated encoder.


    def fit_all(self, input_data, labels, filepath, epochs=1, batch_size=32, patience=3, pre_ep=0):
        """
        Optimized training function for ContentBasedRecommender using tf.data for efficient input handling.
        Includes early stopping and model checkpointing.

        Args:
            input_data: Input features for training.
            labels: Array of labels corresponding to input data.
            filepath: Path to save model checkpoints.
            epochs: Number of epochs to train the model.
            batch_size: Batch size for training.
            patience: Number of epochs to wait for improvement before stopping training early.
            pre_ep: Number of pre-existing epochs to continue from (useful for resumed training).
        """
        import gc
        import tensorflow as tf
        import numpy as np
        from tqdm import tqdm
        import matplotlib.pyplot as plt

        # Create TensorFlow dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((input_data, labels))
        train_dataset = (
            train_dataset.shuffle(buffer_size=1000)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        # Initialize variables for early stopping
        best_loss = np.inf
        patience_counter = 0

        # List to store epoch losses
        train_losses = []

        # Training loop
        for epoch in tqdm(range(epochs), desc="Epochs", unit="epoch"):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # Variable to track epoch loss
            epoch_train_loss = []

            for (X_batch, y_batch) in tqdm(train_dataset, desc="Training", unit="batch", leave=False):
                # Train on batch
                start_time = time.time()
                loss = self.model.train_on_batch(X_batch, y_batch)
                end_time = time.time()
                print(f"Batch training took {end_time - start_time:.2f} seconds.")

                # Track batch loss
                epoch_train_loss.append(loss)

            # Compute average loss for the epoch
            train_loss = np.mean(epoch_train_loss)
            train_losses.append(train_loss)
            print(f"Epoch {epoch + 1} - Loss: {train_loss:.4f}")

            # Early stopping logic
            if train_loss < best_loss:
                best_loss = train_loss
                patience_counter = 0
                # Save model weights for the current epoch
                self.model.save_weights(f"{filepath}/model/ep_{epoch + 1 + pre_ep}.weights.h5")
                self.scorer.save_weights(f"{filepath}/scorer/ep_{epoch + 1 + pre_ep}.weights.h5")
                print(f"Saved model weights for epoch {epoch + 1}.")
            else:
                patience_counter += 1
                print(f"No improvement. Patience counter: {patience_counter}/{patience}.")
                if patience_counter >= patience:
                    print(f"Stopping early at epoch {epoch + 1}. Best loss: {best_loss:.4f}")
                    break

            # Clear memory after each epoch
            gc.collect()
            tf.keras.backend.clear_session()

        # Plot training loss
        plt.figure(figsize=(12, 5))
        plt.plot(train_losses, marker='o', linestyle='-', label='Training Loss')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{filepath}/loss_curve.png")
        plt.show()

    def predict(self, input_data, batch_size=32):
        """
        Predict method for ContentBasedRecommender.
        Processes input data in batches and returns predictions.

        Args:
            input_data: Input data for prediction (e.g., test articles).
            batch_size: Batch size for processing.

        Returns:
            numpy.ndarray: Predictions for the input data.
        """
        import numpy as np
        from tqdm import tqdm

        # Calculate the number of batches
        num_batches = len(input_data) // batch_size

        # Initialize a list to store predictions
        predictions = []

        # Process data in batches
        for batch_index in tqdm(range(num_batches), desc="Batches", unit="batch"):
            # Create each batch
            X_batch = input_data[batch_index * batch_size:(batch_index + 1) * batch_size]

            # Predict on batch
            batch_predictions = self.scorer.predict_on_batch(X_batch)

            # Store batch predictions
            predictions.append(batch_predictions)

        # Handle the last batch if the total data is not perfectly divisible by batch_size
        if len(input_data) % batch_size != 0:
            X_batch = input_data[num_batches * batch_size:]
            if len(X_batch) > 0:
                batch_predictions = self.scorer.predict_on_batch(X_batch)
                predictions.append(batch_predictions)

        # Concatenate all batch predictions into a single array
        predictions = np.concatenate(predictions, axis=0)

        return predictions


