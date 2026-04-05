# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.


import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt


import tensorflow.keras as keras
from tensorflow.keras import layers


from src.models.base_model import BaseModel
from src.models.layers import PersonalizedAttentivePooling

__all__ = ["NPAModel"]


class NPAModel(BaseModel):
    """NPA model(Neural News Recommendation with Attentive Multi-View Learning)

    Chuhan Wu, Fangzhao Wu, Mingxiao An, Jianqiang Huang, Yongfeng Huang and Xing Xie:
    NPA: Neural News Recommendation with Personalized Attention, KDD 2019, ADS track.

    Attributes:
        word2vec_embedding (numpy.ndarray): Pretrained word embedding matrix.
        hparam (object): Global hyper-parameters.
    """

    def __init__(self, hparams,user_size, seed=None):
        """Initialization steps for MANL.
        Compared with the BaseModel, NPA need word embedding.
        After creating word embedding matrix, BaseModel's __init__ method will be called.

        Args:
            hparams (object): Global hyper-parameters. Some key setttings such as filter_num are there.
            iterator_creator_train (object): NPA data loader class for train data.
            iterator_creator_test (object): NPA data loader class for test and validation data
        """

        self.word2vec_embedding = self._init_embedding(hparams.wordEmb_file)
        self.hparam = hparams
        self.user_size= user_size

        super().__init__(hparams, seed=seed)



    def _build_graph(self):
        """Build NPA model and scorer.

        Returns:
            object: a model used to train.
            object: a model used to evaluate and inference.
        """

        model, scorer = self._build_npa()
        return model, scorer

    def _build_userencoder(self, titleencoder, user_embedding_layer):
        """The main function to create user encoder of NPA.

        Args:
            titleencoder (object): the news encoder of NPA.

        Return:
            object: the user encoder of NPA.
        """
        hparams = self.hparams

        his_input_title = keras.Input(
            shape=(hparams.his_size, hparams.title_size), dtype="int32"
        )
        user_indexes = keras.Input(shape=(1,), dtype="int32")

        nuser_id = layers.Reshape((1, 1))(user_indexes)
        repeat_uids = layers.Concatenate(axis=-2)([nuser_id] * hparams.his_size)
        his_title_uid = layers.Concatenate(axis=-1)([his_input_title, repeat_uids])

        click_title_presents = layers.TimeDistributed(titleencoder)(his_title_uid)

        u_emb = layers.Reshape((hparams.user_emb_dim,))(
            user_embedding_layer(user_indexes)
        )
        user_present = PersonalizedAttentivePooling(
            hparams.his_size,
            hparams.filter_num,
            hparams.attention_hidden_dim,
            seed=self.seed,
        )([click_title_presents, layers.Dense(hparams.attention_hidden_dim)(u_emb)])

        model = keras.Model(
            [his_input_title, user_indexes], user_present, name="user_encoder"
        )
        return model

    def _build_newsencoder(self, embedding_layer, user_embedding_layer):
        """The main function to create news encoder of NPA.

        Args:
            embedding_layer (object): a word embedding layer.

        Return:
            object: the news encoder of NPA.
        """
        hparams = self.hparams
        sequence_title_uindex = keras.Input(
            shape=(hparams.title_size + 1,), dtype="int32"
        )

        sequences_input_title = layers.Lambda(lambda x: x[:, : hparams.title_size])(
            sequence_title_uindex
        )
        user_index = layers.Lambda(lambda x: x[:, hparams.title_size :])(
            sequence_title_uindex
        )

        u_emb = layers.Reshape((hparams.user_emb_dim,))(
            user_embedding_layer(user_index)
        )
        embedded_sequences_title = embedding_layer(sequences_input_title)

        y = layers.Dropout(hparams.dropout)(embedded_sequences_title)
        y = layers.Conv1D(
            hparams.filter_num,
            hparams.window_size,
            activation=hparams.cnn_activation,
            padding="same",
            bias_initializer=keras.initializers.Zeros(),
            kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
        )(y)
        y = layers.Dropout(hparams.dropout)(y)

        pred_title = PersonalizedAttentivePooling(
            hparams.title_size,
            hparams.filter_num,
            hparams.attention_hidden_dim,
            seed=self.seed,
        )([y, layers.Dense(hparams.attention_hidden_dim)(u_emb)])

        # pred_title = Reshape((1, feature_size))(pred_title)
        model = keras.Model(sequence_title_uindex, pred_title, name="news_encoder")
        return model

    def _build_npa(self):
        """The main function to create NPA's logic. The core of NPA
        is a user encoder and a news encoder.

        Returns:
            object: a model used to train.
            object: a model used to evaluate and predict.
        """
        hparams = self.hparams

        his_input_title = keras.Input(
            shape=(hparams.his_size, hparams.title_size), dtype="int32"
        )
        pred_input_title = keras.Input(
            shape=(hparams.npratio + 1, hparams.title_size), dtype="int32"
        )
        pred_input_title_one = keras.Input(
            shape=(
                1,
                hparams.title_size,
            ),
            dtype="int32",
        )
        pred_title_one_reshape = layers.Reshape((hparams.title_size,))(
            pred_input_title_one
        )
        user_indexes = keras.Input(shape=(1,), dtype="int32")

        nuser_index = layers.Reshape((1, 1))(user_indexes)
        repeat_uindex = layers.Concatenate(axis=-2)(
            [nuser_index] * (hparams.npratio + 1)
        )
        pred_title_uindex = layers.Concatenate(axis=-1)(
            [pred_input_title, repeat_uindex]
        )
        pred_title_uindex_one = layers.Concatenate()(
            [pred_title_one_reshape, user_indexes]
        )

        embedding_layer = layers.Embedding(
            self.word2vec_embedding.shape[0],
            hparams.word_emb_dim,
            weights=[self.word2vec_embedding],
            trainable=True,
        )

        user_embedding_layer = layers.Embedding(
            self.user_size,
            hparams.user_emb_dim,
            trainable=True,
            embeddings_initializer="zeros",
        )

        titleencoder = self._build_newsencoder(embedding_layer, user_embedding_layer)
        userencoder = self._build_userencoder(titleencoder, user_embedding_layer)
        newsencoder = titleencoder

        user_present = userencoder([his_input_title, user_indexes])

        news_present = layers.TimeDistributed(newsencoder)(pred_title_uindex)
        news_present_one = newsencoder(pred_title_uindex_one)

        preds = layers.Dot(axes=-1)([news_present, user_present])
        preds = layers.Activation(activation="softmax")(preds)

        pred_one = layers.Dot(axes=-1)([news_present_one, user_present])
        pred_one = layers.Activation(activation="sigmoid")(pred_one)

        model = keras.Model([user_indexes, his_input_title, pred_input_title], preds)
        scorer = keras.Model(
            [user_indexes, his_input_title, pred_input_title_one], pred_one
        )

        return model, scorer

    def fit(self, input_data, user_indexes, labels, filepath, epochs=1, batch_size=32, patience=3, pre_ep=0):
        """
        Optimized training function for NPA using tf.data for efficient input handling.
        Includes early stopping and model checkpointing.

        Args:
            input_data: Tuple (X_train_history, X_train_candidates) representing input features.
            user_indexes: Array of user indexes corresponding to input data.
            labels: Array of labels corresponding to input data.
            filepath: Path to save model checkpoints.
            epochs: Number of epochs to train the model.
            batch_size: Batch size for training.
            patience: Number of epochs to wait for improvement before stopping training early.
        """
        import gc
        import tensorflow as tf
        import numpy as np
        from tqdm import tqdm
        import matplotlib.pyplot as plt

        # Unpack input data
        X_train_history, X_train_candidates = input_data

        # Create TensorFlow dataset
        train_dataset = tf.data.Dataset.from_tensor_slices(
            ((user_indexes, X_train_history, X_train_candidates), labels)
        )
        train_dataset = (
            train_dataset.shuffle(buffer_size=1000)
            .batch(batch_size)
            .prefetch(1)
        )

        # Initialize variables for early stopping
        best_loss = np.inf
        patience_counter = 0

        # Initialize list to store epoch loss
        train_losses = []

        # Training loop
        for epoch in tqdm(range(epochs), desc="Epochs", unit="epoch"):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # Initialize variable for tracking batch loss
            epoch_train_loss = []

            for (X_batch, y_batch) in tqdm(train_dataset, desc="Training", unit="batch", leave=False):
                # Train on batch
                start_time = time.time()
                loss = self.model.train_on_batch(X_batch, y_batch)
                end_time = time.time()
                print(f"Training took {end_time - start_time:.2f} seconds.")

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
                self.model.save_weights(f"{filepath}model/ep_{epoch + 1 + pre_ep}.weights.h5")
                self.scorer.save_weights(f"{filepath}scorer/ep_{epoch + 1 + pre_ep}.weights.h5")
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



    def predict(self, input_data, user_indexes, batch_size=32):
        # Assuming input_data is a tuple (X_test_history, X_test_candidates)
        X_test_history, X_test_candidates = input_data

        # Calculate the number of batches
        num_batches = len(X_test_history) // batch_size

        # Initialize a list to store predictions
        predictions = []

        # Use tqdm to display progress for batches
        for batch_index in tqdm(range(num_batches), desc="Batches", unit="batch"):
            # Create each batch
            X_batch_history = X_test_history[batch_index * batch_size:(batch_index + 1) * batch_size]
            X_batch_candidates = X_test_candidates[batch_index * batch_size:(batch_index + 1) * batch_size]
            X_batch_user_indexes = user_indexes[batch_index * batch_size:(batch_index + 1) * batch_size]

            # Predict on batch
            batch_predictions = self.scorer.predict_on_batch(
                [X_batch_user_indexes, X_batch_history, X_batch_candidates])

            # Store batch predictions
            predictions.append(batch_predictions)

        # Handle the last batch if the total data is not perfectly divisible by batch_size
        if len(X_test_history) % batch_size != 0:
            X_batch_history = X_test_history[num_batches * batch_size:]
            X_batch_candidates = X_test_candidates[num_batches * batch_size:]
            X_batch_user_indexes = user_indexes[num_batches * batch_size:]
            if len(X_batch_history) > 0:
                batch_predictions = self.scorer.predict_on_batch(
                    [X_batch_user_indexes, X_batch_history, X_batch_candidates])
                predictions.append(batch_predictions)

        # Concatenate all batch predictions into a single array
        predictions = np.concatenate(predictions, axis=0)

        return predictions


