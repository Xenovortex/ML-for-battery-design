from typing import Type

import tensorflow as tf
from bayesflow.default_settings import MetaDictSetting
from tensorflow.keras.layers import LSTM, Dense, Flatten
from tensorflow.keras.models import Sequential


class FC_network(tf.keras.Model):
    """Implements a fully-connected network with keras.

    Attributes:
        FC (tf.keras.Model): fully-connected network architecture
    """

    def __init__(self, meta: Type[MetaDictSetting]) -> None:
        """Initializes :class:FC_network

        Args:
            meta (bayesflow.MetaDictSetting): contains data to construct network architecture
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
        TODO
    """

    def __init__(self, meta: Type[MetaDictSetting]) -> None:
        """Initializes :class:LSTM_network

        Args:
            meta (Type[MetaDictSetting]): contains data to construct network architecture
        """
        super(LSTM_network, self).__init__()

        self.LSTM = Sequential(
            [LSTM(unit, return_sequences=True) for unit in meta["units"][:-1]]
            + [LSTM(meta["units"][-1], return_sequences=False)]
            + [Dense(meta["summary_dim"], activation="sigmoid")]
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Performs the forward pass of the model

        Args:
            x (tf.Tensor): input data of shape (batch_size, max_time_iter, (nr,) num_features)

        Returns:
            out (tf.Tensor): output of shape (batch_size, summary_dim)
        """
        out = self.LSTM(x)
        return out
