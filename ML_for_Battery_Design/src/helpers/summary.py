from typing import Type

import tensorflow as tf
from bayesflow.default_settings import MetaDictSetting
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Conv2D,
    Dense,
    Flatten,
    GlobalAveragePooling2D,
    MaxPool2D,
)


class FC_Network(tf.keras.Model):
    """Implements a fully-connected network with keras.

    Attributes:
        FC (tf.keras.Model): fully-connected network architecture
    """

    def __init__(self, meta: Type[MetaDictSetting]) -> None:
        """Initializes :class:FC_network

        Args:
            meta (bayesflow.MetaDictSetting): contains settings to construct network architecture
        """
        super(FC_Network, self).__init__()

        self.FC = Sequential()
        self.FC.add(Flatten())
        if meta["units"] is not None:
            for unit in meta["units"]:
                self.FC.add(Dense(unit, activation=meta["activation"]))
        self.FC.add(Dense(meta["summary_dim"], activation="sigmoid"))

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Performs the forward pass of the model

        Args:
            x (tf.Tensor): input data of shape (batch_size, max_time_iter, (nr,) num_features)

        Returns:
            out (tf.Tensor): output of shape (batch_size, summary_dim)
        """
        out = self.FC(x)
        return out


class LSTM_Network(tf.keras.Model):
    """Implements a long short-term memory network architecture

    Attributes:
        LSTM (tf.keras.Model): long short-term memory network architecture
    """

    def __init__(self, meta: Type[MetaDictSetting]) -> None:
        """Initializes :class:LSTM_network

        Args:
            meta (Type[MetaDictSetting]): contains settings to construct network architecture
        """
        super(LSTM_Network, self).__init__()

        self.LSTM = Sequential()
        if meta["lstm_units"] is not None:
            if len(meta["lstm_units"]) > 1:
                for unit in meta["lstm_units"][:-1]:
                    self.LSTM.add(LSTM(unit, return_sequences=True))
            self.LSTM.add(LSTM(meta["lstm_units"][-1], return_sequences=False))
        else:
            self.LSTM.add(Flatten())
        if meta["fc_units"] is not None:
            for unit in meta["fc_units"]:
                self.LSTM.add(Dense(unit, activation=meta["fc_activation"]))
        self.LSTM.add(Dense(meta["summary_dim"], activation="sigmoid"))

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Performs the forward pass of the model

        Args:
            x (tf.Tensor): input data of shape (batch_size, max_time_iter, num_features)

        Returns:
            out (tf.Tensor): output of shape (batch_size, summary_dim)
        """
        out = self.LSTM(x)
        return out


class CNN_Network(tf.keras.Model):
    """Implements a 2D convolutional neural network (VGG inspired)

    Attributes:
        CNN (tf.keras.Model): 2D convolutional neural network architecture
    """

    def __init__(self, meta: Type[MetaDictSetting]) -> None:
        """Initializes :class:CNN_network

        Args:
            meta (Type[MetaDictSetting]): contains settings to construct network architecture
        """
        super(CNN_Network, self).__init__()

        time_pool_size = meta["pool_time"] + 1
        space_pool_size = meta["pool_space"] + 1

        self.num_cnn_blocks = (
            len(meta["num_filters"]) if meta["num_filters"] is not None else 0
        )
        self.min_input_time_dim = time_pool_size**self.num_cnn_blocks
        self.min_input_space_dim = space_pool_size**self.num_cnn_blocks

        self.CNN = Sequential()
        if meta["num_filters"] is not None:
            for num_filters in meta["num_filters"]:
                self.CNN.add(
                    Conv2D(
                        num_filters,
                        kernel_size=3,
                        strides=1,
                        padding="same",
                        activation=meta["cnn_activation"],
                    )
                )
                self.CNN.add(
                    Conv2D(
                        num_filters,
                        kernel_size=3,
                        strides=1,
                        padding="same",
                        activation=meta["cnn_activation"],
                    )
                )
                self.CNN.add(MaxPool2D(pool_size=(time_pool_size, space_pool_size)))
            self.CNN.add(GlobalAveragePooling2D())
        else:
            self.CNN.add(Flatten())
        if meta["units"] is not None:
            for unit in meta["units"]:
                self.CNN.add(Dense(unit, activation=meta["fc_activation"]))
        self.CNN.add(Dense(meta["summary_dim"], activation="sigmoid"))

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Performs the forward pass of the model

        Args:
            x (tf.Tensor): input data of shape (batch_size, max_time_iter, nr, num_features)

        Returns:
            out (tf.Tensor): output of shape (batch_size, summary_dim)
        """
        if self.num_cnn_blocks is not None:
            if x.shape[1] < self.min_input_time_dim:
                raise ValueError(
                    "{} - call: input x has shape {}, but applying max_pool {} times on time dimension {} leads to output shape {}".format(
                        self.__class__.__name__,
                        x.shape,
                        self.num_cnn_blocks,
                        x.shape[1],
                        (
                            x.shape[0],
                            x.shape[1] // self.min_input_time_dim,
                            x.shape[2] // self.min_input_space_dim,
                            x.shape[3],
                        ),
                    )
                )
            if x.shape[2] < self.min_input_space_dim:
                raise ValueError(
                    "{} - call: input x has shape {}, but applying max_pool {} times on space dimension {} leads to output shape {}".format(
                        self.__class__.__name__,
                        x.shape,
                        self.num_cnn_blocks,
                        x.shape[2],
                        (
                            x.shape[0],
                            x.shape[1] // self.min_input_time_dim,
                            x.shape[2] // self.min_input_space_dim,
                            x.shape[3],
                        ),
                    )
                )

        out = self.CNN(x)
        return out


summary_collection = {
    "FC": FC_Network,
    "LSTM": LSTM_Network,
    "CNN": CNN_Network,
}
