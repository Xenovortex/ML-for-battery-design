from typing import Type

import tensorflow as tf
from bayesflow.default_settings import MetaDictSetting
from tensorflow.keras.layers import (
    LSTM,
    Conv2D,
    Dense,
    Flatten,
    GlobalAveragePooling2D,
    MaxPool2D,
)
from tensorflow.keras.models import Sequential


class FC_network(tf.keras.Model):
    """Implements a fully-connected network with keras.

    Attributes:
        FC (tf.keras.Model): fully-connected network architecture
    """

    def __init__(self, meta: Type[MetaDictSetting]) -> None:
        """Initializes :class:FC_network

        Args:
            meta (bayesflow.MetaDictSetting): contains settings to construct network architecture
        """
        super(FC_network, self).__init__()

        self.FC = Sequential(
            [Flatten()]
            + [Dense(unit, activation=meta["activation"]) for unit in meta["units"]]
            + [Dense(meta["summary_dim"], activation="sigmoid")]
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Performs the forward pass of the model

        Args:
            x (tf.Tensor): input data of shape (batch_size, max_time_iter, (nr,) num_features)

        Returns:
            out (tf.Tensor): output of shape (batch_size, summary_dim)
        """
        out = self.FC(x)
        return out


class LSTM_network(tf.keras.Model):
    """Implements a long short-term memory network architecture

    Attributes:
        LSTM (tf.keras.Model): long short-term memory network architecture
    """

    def __init__(self, meta: Type[MetaDictSetting]) -> None:
        """Initializes :class:LSTM_network

        Args:
            meta (Type[MetaDictSetting]): contains settings to construct network architecture
        """
        super(LSTM_network, self).__init__()

        self.LSTM = Sequential(
            [LSTM(unit, return_sequences=True) for unit in meta["lstm_units"][:-1]]
            + [LSTM(meta["lstm_units"][-1], return_sequences=False)]
            + [
                Dense(unit, activation=meta["fc_activation"])
                for unit in meta["fc_units"]
            ]
            + [Dense(meta["summary_dim"], activation="sigmoid")]
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Performs the forward pass of the model

        Args:
            x (tf.Tensor): input data of shape (batch_size, max_time_iter, num_features)

        Returns:
            out (tf.Tensor): output of shape (batch_size, summary_dim)
        """
        out = self.LSTM(x)
        return out


class CNN_network(tf.keras.Model):
    """Implements a 2D convolutional neural network

    Attributes:
        CNN (tf.keras.Model): 2D convolutional neural network architecture
    """

    def __init__(self, meta: Type[MetaDictSetting]) -> None:
        """Initializes :class:CNN_network

        Args:
            meta (Type[MetaDictSetting]): contains settings to construct network architecture
        """
        super(CNN_network, self).__init__()

        time_pool_size = 2 if meta["pool_time"] else 1
        space_pool_size = 2 if meta["pool_space"] else 1

        CNN_base = []
        for num_filters, kernel_size, stride in zip(
            meta["num_filters"], meta["kernel_size"], meta["stride"]
        ):
            CNN_base += [
                Conv2D(
                    num_filters,
                    kernel_size,
                    stride,
                    padding="same",
                    activation=meta["cnn_activation"],
                ),
                Conv2D(
                    num_filters,
                    kernel_size,
                    stride,
                    padding="same",
                    activation=meta["cnn_activation"],
                ),
                MaxPool2D(
                    pool_size=(time_pool_size, space_pool_size),
                    stride=(time_pool_size, space_pool_size),
                    padding="same",
                ),
            ]
        self.CNN = Sequential(
            CNN_base
            + [GlobalAveragePooling2D()]
            + [Dense(unit, activation=meta["fc_activation"]) for unit in meta["units"]]
            + [Dense(meta["summary_dim"], activation="sigmoid")]
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Performs the forward pass of the model

        Args:
            x (tf.Tensor): input data of shape (batch_size, max_time_iter, nr, num_features)

        Returns:
            out (tf.Tensor): output of shape (batch_size, summary_dim)
        """
        out = self.CNN(x)
        return out
