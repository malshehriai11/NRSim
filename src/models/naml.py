# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import layers
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import gc
import tensorflow as tf




from src.models.base_model import BaseModel
from src.models.layers import AttLayer2

__all__ = ["NAMLModel"]


class NAMLModel(BaseModel):
    """NAML model(Neural News Recommendation with Attentive Multi-View Learning)

    Chuhan Wu, Fangzhao Wu, Mingxiao An, Jianqiang Huang, Yongfeng Huang and Xing Xie,
    Neural News Recommendation with Attentive Multi-View Learning, IJCAI 2019

    Attributes:
        word2vec_embedding (numpy.ndarray): Pretrained word embedding matrix.
        hparam (object): Global hyper-parameters.
    """

    def __init__(self, hparams, seed=None):
        """Initialization steps for NAML.
        Compared with the BaseModel, NAML need word embedding.
        After creating word embedding matrix, BaseModel's __init__ method will be called.

        Args:
            hparams (object): Global hyper-parameters. Some key setttings such as filter_num are there.
            iterator_creator_train (object): NAML data loader class for train data.
            iterator_creator_test (object): NAML data loader class for test and validation data
        """

        self.word2vec_embedding = self._init_embedding(hparams.wordEmb_file)
        self.hparam = hparams

        super().__init__(hparams, seed=seed)

    def _get_input_label_from_iter(self, batch_data):
        input_feat = [
            batch_data["clicked_title_batch"],
            batch_data["clicked_ab_batch"],
            batch_data["clicked_vert_batch"],
            batch_data["clicked_subvert_batch"],
            batch_data["candidate_title_batch"],
            batch_data["candidate_ab_batch"],
            batch_data["candidate_vert_batch"],
            batch_data["candidate_subvert_batch"],
        ]
        input_label = batch_data["labels"]
        return input_feat, input_label

    def _get_user_feature_from_iter(self, batch_data):
        """get input of user encoder
        Args:
            batch_data: input batch data from user iterator

        Returns:
            numpy.ndarray: input user feature (clicked title batch)
        """
        input_feature = [
            batch_data["clicked_title_batch"],
            batch_data["clicked_ab_batch"],
            batch_data["clicked_vert_batch"],
            batch_data["clicked_subvert_batch"],
        ]
        input_feature = np.concatenate(input_feature, axis=-1)
        return input_feature

    def _get_news_feature_from_iter(self, batch_data):
        """get input of news encoder
        Args:
            batch_data: input batch data from news iterator

        Returns:
            numpy.ndarray: input news feature (candidate title batch)
        """
        input_feature = [
            batch_data["candidate_title_batch"],
            batch_data["candidate_ab_batch"],
            batch_data["candidate_vert_batch"],
            batch_data["candidate_subvert_batch"],
        ]
        input_feature = np.concatenate(input_feature, axis=-1)
        return input_feature

    def _build_graph(self):
        """Build NAML model and scorer.

        Returns:
            object: a model used to train.
            object: a model used to evaluate and inference.
        """

        model, scorer = self._build_naml()
        return model, scorer

    def _build_userencoder(self, newsencoder):
        """The main function to create user encoder of NAML.

        Args:
            newsencoder (object): the news encoder of NAML.

        Return:
            object: the user encoder of NAML.
        """
        hparams = self.hparams
        his_input_title_body_ctg = keras.Input(
            shape=(hparams.his_size, hparams.title_size + hparams.body_size + 1),
            dtype="int32",
        )

        click_news_presents = layers.TimeDistributed(newsencoder)(
            his_input_title_body_ctg
        )
        user_present = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(
            click_news_presents
        )

        model = keras.Model(
            his_input_title_body_ctg, user_present, name="user_encoder"
        )
        return model

    def _build_newsencoder(self, embedding_layer):
        """The main function to create news encoder of NAML.
        news encoder in composed of title encoder, body encoder, vert encoder and subvert encoder

        Args:
            embedding_layer (object): a word embedding layer.

        Return:
            object: the news encoder of NAML.
        """
        hparams = self.hparams
        input_title_body_ctgs = keras.Input(
            shape=(hparams.title_size + hparams.body_size + 1,), dtype="int32"
        )

        sequences_input_title = layers.Lambda(lambda x: x[:, : hparams.title_size])(
            input_title_body_ctgs
        )
        sequences_input_body = layers.Lambda(
            lambda x: x[:, hparams.title_size : hparams.title_size + hparams.body_size]
        )(input_title_body_ctgs)
        input_ctg = layers.Lambda(
            lambda x: x[
                :,
                hparams.title_size
                + hparams.body_size : hparams.title_size
                + hparams.body_size
                + 1,
            ]
        )(input_title_body_ctgs)


        title_repr = self._build_titleencoder(embedding_layer)(sequences_input_title)
        body_repr = self._build_bodyencoder(embedding_layer)(sequences_input_body)
        ctg_repr = self._build_ctgencoder()(input_ctg)

        concate_repr = layers.Concatenate(axis=-2)(
            [title_repr, body_repr, ctg_repr]
        )
        news_repr = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(
            concate_repr
        )

        model = keras.Model(input_title_body_ctgs, news_repr, name="news_encoder")
        return model

    def _build_titleencoder(self, embedding_layer):
        """build title encoder of NAML news encoder.

        Args:
            embedding_layer (object): a word embedding layer.

        Return:
            object: the title encoder of NAML.
        """
        hparams = self.hparams
        sequences_input_title = keras.Input(shape=(hparams.title_size,), dtype="int32")
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
        pred_title = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(y)
        pred_title = layers.Reshape((1, hparams.filter_num))(pred_title)

        model = keras.Model(sequences_input_title, pred_title, name="title_encoder")
        return model

    def _build_bodyencoder(self, embedding_layer):
        """build body encoder of NAML news encoder.

        Args:
            embedding_layer (object): a word embedding layer.

        Return:
            object: the body encoder of NAML.
        """
        hparams = self.hparams
        sequences_input_body = keras.Input(shape=(hparams.body_size,), dtype="int32")
        embedded_sequences_body = embedding_layer(sequences_input_body)

        y = layers.Dropout(hparams.dropout)(embedded_sequences_body)
        y = layers.Conv1D(
            hparams.filter_num,
            hparams.window_size,
            activation=hparams.cnn_activation,
            padding="same",
            bias_initializer=keras.initializers.Zeros(),
            kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
        )(y)
        y = layers.Dropout(hparams.dropout)(y)
        pred_body = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(y)
        pred_body = layers.Reshape((1, hparams.filter_num))(pred_body)

        model = keras.Model(sequences_input_body, pred_body, name="body_encoder")
        return model

    def _build_ctgencoder(self):
        """build ctg encoder of NAML news encoder.

        Return:
            object: the ctg encoder of NAML.
        """
        hparams = self.hparams
        input_ctg = keras.Input(shape=(1,), dtype="int32")

        ctg_embedding = layers.Embedding(
            hparams.ctg_num, hparams.ctg_emb_dim, trainable=True
        )

        ctg_emb = ctg_embedding(input_ctg)
        pred_ctg = layers.Dense(
            hparams.filter_num,
            activation=hparams.dense_activation,
            bias_initializer=keras.initializers.Zeros(),
            kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
        )(ctg_emb)
        pred_ctg = layers.Reshape((1, hparams.filter_num))(pred_ctg)

        model = keras.Model(input_ctg, pred_ctg, name="ctg_encoder")
        return model


    def _build_naml(self):
        """The main function to create NAML's logic. The core of NAML
        is a user encoder and a news encoder.

        Returns:
            object: a model used to train.
            object: a model used to evaluate and predict.
        """
        hparams = self.hparams

        his_input_title = keras.Input(
            shape=(hparams.his_size, hparams.title_size), dtype="int32"
        )
        his_input_body = keras.Input(
            shape=(hparams.his_size, hparams.body_size), dtype="int32"
        )
        his_input_ctg = keras.Input(shape=(hparams.his_size, 1), dtype="int32")

        pred_input_title = keras.Input(
            shape=(hparams.npratio + 1, hparams.title_size), dtype="int32"
        )
        pred_input_body = keras.Input(
            shape=(hparams.npratio + 1, hparams.body_size), dtype="int32"
        )
        pred_input_ctg = keras.Input(shape=(hparams.npratio + 1, 1), dtype="int32")

        pred_input_title_one = keras.Input(
            shape=(
                1,
                hparams.title_size,
            ),
            dtype="int32",
        )
        pred_input_body_one = keras.Input(
            shape=(
                1,
                hparams.body_size,
            ),
            dtype="int32",
        )
        pred_input_ctg_one = keras.Input(shape=(1, 1), dtype="int32")

        his_title_body_ctg = layers.Concatenate(axis=-1)(
            [his_input_title, his_input_body, his_input_ctg]
        )

        pred_title_body_ctg = layers.Concatenate(axis=-1)(
            [pred_input_title, pred_input_body, pred_input_ctg]
        )

        pred_title_body_ctg_one = layers.Concatenate(axis=-1)(
            [
                pred_input_title_one,
                pred_input_body_one,
                pred_input_ctg_one
            ]
        )
        pred_title_body_ctg_one = layers.Reshape((-1,))(pred_title_body_ctg_one)

        embedding_layer = layers.Embedding(
            self.word2vec_embedding.shape[0],
            hparams.word_emb_dim,
            weights=[self.word2vec_embedding],
            trainable=True,
        )

        self.newsencoder = self._build_newsencoder(embedding_layer)
        self.userencoder = self._build_userencoder(self.newsencoder)

        user_present = self.userencoder(his_title_body_ctg)
        news_present = layers.TimeDistributed(self.newsencoder)(pred_title_body_ctg)
        news_present_one = self.newsencoder(pred_title_body_ctg_one)

        preds = layers.Dot(axes=-1)([news_present, user_present])
        preds = layers.Activation(activation="softmax")(preds)

        pred_one = layers.Dot(axes=-1)([news_present_one, user_present])
        pred_one = layers.Activation(activation="sigmoid")(pred_one)

        model = keras.Model(
            [
                his_input_title,
                his_input_body,
                his_input_ctg,
                pred_input_title,
                pred_input_body,
                pred_input_ctg,
            ],
            preds,
        )

        scorer = keras.Model(
            [
                his_input_title,
                his_input_body,
                his_input_ctg,
                pred_input_title_one,
                pred_input_body_one,
                pred_input_ctg_one,
            ],
            pred_one,
        )

        return model, scorer

    def fit(self, input_data, labels, filepath, epochs=1, batch_size=32, patience=3, pre_ep=0):
        """
        Optimized training function for NAML model with efficient memory management.

        Args:
            input_data: List of six arrays [X1, X2, X3, X4, X5, X6] representing input features.
            labels: Array of labels corresponding to input data.
            filepath: Path to save model checkpoints.
            epochs: Number of epochs to train the model.
            batch_size: Batch size for training.
            patience: Number of epochs to wait for improvement before early stopping.
        """
        # Unpack inputs
        X1, X2, X3, X4, X5, X6 = input_data
        labels = np.array(labels)

        # print(f"X1 shape: {np.shape(X1)}")
        # print(f"X2 shape: {np.shape(X2)}")
        # print(f"X3 shape: {np.shape(X3)}")
        # print(f"X4 shape: {np.shape(X4)}")
        # print(f"X5 shape: {np.shape(X5)}")
        # print(f"X6 shape: {np.shape(X6)}")
        # print(f"Labels shape: {np.shape(labels)}")


        # Create TensorFlow dataset
        train_dataset = tf.data.Dataset.from_tensor_slices(((X1, X2, X3, X4, X5, X6), labels))
        train_dataset = train_dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(1)

        # Initialize variables for tracking loss and early stopping
        train_losses = []
        best_loss = np.inf
        patience_counter = 0

        # Training loop
        for epoch in tqdm(range(epochs), desc="Epochs", unit="epoch"):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # Training step
            epoch_train_loss = 0
            for X_batch, y_batch in tqdm(train_dataset, desc="Training", unit="batch", leave=False):
                start_time = time.time()
                loss = self.model.train_on_batch(X_batch, y_batch)
                end_time = time.time()
                print(f"Batch Training Time: {end_time - start_time:.4f} seconds")
                epoch_train_loss += loss

            # Compute average loss for the epoch
            train_loss = epoch_train_loss / len(train_dataset)
            train_losses.append(train_loss)

            # Log epoch statistics
            print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}")


            # Save weights if training loss improves
            if train_loss < best_loss:
                best_loss = train_loss
                patience_counter = 0  # Reset patience counter
                # Save model weights for the current epoch
                self.model.save_weights(f"{filepath}model/ep_{epoch + 1 + pre_ep}.weights.h5")
                self.scorer.save_weights(f"{filepath}scorer/ep_{epoch + 1+ pre_ep}.weights.h5")
                print(f"Saved best model at epoch {epoch + 1}")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}. Best Train Loss: {best_loss:.4f}")
                break

            # Clear memory after each epoch
            gc.collect()
            tf.keras.backend.clear_session()

        # Plot training loss
        plt.figure(figsize=(12, 5))
        plt.plot(train_losses, label="Training Loss", marker="o", linestyle="-")
        plt.title("Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{filepath}/loss_curve.png", dpi=300)
        plt.show()

        return train_losses


    def predict(self, input_data, batch_size=32):

        # Simulated data
        X_test = input_data  # X_test is a list of two ndarrays

        # Calculate the number of batches
        num_batches = len(X_test[0]) // batch_size

        # Initialize a list to store predictions
        predictions = []

        # Use tqdm to display progress for batches
        for batch_index in tqdm(range(num_batches), desc="Batches", unit="batch"):
            # Batch both inputs
            X_batch_1 = X_test[0][batch_index * batch_size:(batch_index + 1) * batch_size]
            X_batch_2 = X_test[1][batch_index * batch_size:(batch_index + 1) * batch_size]
            X_batch_3 = X_test[2][batch_index * batch_size:(batch_index + 1) * batch_size]
            X_batch_4 = X_test[3][batch_index * batch_size:(batch_index + 1) * batch_size]
            X_batch_5 = X_test[4][batch_index * batch_size:(batch_index + 1) * batch_size]
            X_batch_6 = X_test[5][batch_index * batch_size:(batch_index + 1) * batch_size]

            # Predict on batch
            # Ensure eager execution is enabled
            # tf.config.run_functions_eagerly(True)
            # print("Eager execution enabled:", tf.executing_eagerly())

            batch_predictions = self.scorer.predict_on_batch([X_batch_1, X_batch_2,X_batch_3, X_batch_4, X_batch_5, X_batch_6])

            # Store batch predictions
            predictions.append(batch_predictions)
            # print(batch_index)

        # Handle the last batch if the total data is not perfectly divisible by batch_size
        if len(X_test[0]) % batch_size != 0:
            X_batch_1 = X_test[0][num_batches * batch_size:]
            X_batch_2 = X_test[1][num_batches * batch_size:]
            X_batch_3 = X_test[2][num_batches * batch_size:]
            X_batch_4 = X_test[3][num_batches * batch_size:]
            X_batch_5 = X_test[4][num_batches * batch_size:]
            X_batch_6 = X_test[5][num_batches * batch_size:]

            if len(X_batch_1) > 0:
                batch_predictions = self.scorer.predict_on_batch([X_batch_1, X_batch_2,X_batch_3, X_batch_4, X_batch_5, X_batch_6])
                predictions.append(batch_predictions)

        # Concatenate all batch predictions into a single array
        predictions = np.concatenate(predictions, axis=0)

        return predictions
