from typing import Type

import tensorflow as tf
from bayesflow.default_settings import MetaDictSetting
from bayesflow.helper_functions import build_meta_dict
from keras import backend as K
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    LSTM,
    BatchNormalization,
    Conv2D,
    ConvLSTM1D,
    Dense,
    Flatten,
    GlobalAveragePooling1D,
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
        self.FC.add(Dense(meta["summary_dim"]))

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
        self.LSTM.add(Dense(meta["summary_dim"]))

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
        """Initializes :class:CNN_Network

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
        self.CNN.add(Dense(meta["summary_dim"]))

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


class ConvLSTM_Network(tf.keras.Model):
    """Implements a 1D Convolutional LSTM network architecture

    Attributes:
        ConvLSTM (tf.keras.Model): 1D Convolutional LSTM network architecture
    """

    def __init__(self, meta: Type[MetaDictSetting]) -> None:
        """Initializes :class:ConvLSTM_Network

        Args:
            meta (Type[MetaDictSetting]): contains settings to construct network architecture
        """
        super(ConvLSTM_Network, self).__init__()

        time_pool_size = meta["pool_time"] + 1
        space_pool_size = meta["pool_space"] + 1

        self.num_convlstm_blocks = (
            len(meta["num_filters"]) if meta["num_filters"] is not None else 0
        )

        self.min_input_time_dim = time_pool_size**self.num_convlstm_blocks
        self.min_input_space_dim = space_pool_size**self.num_convlstm_blocks

        self.ConvLSTM = Sequential()
        if meta["num_filters"] is not None:
            if len(meta["num_filters"]) > 1:
                for num_filters in meta["num_filters"][:-1]:
                    self.ConvLSTM.add(
                        ConvLSTM1D(
                            num_filters,
                            kernel_size=3,
                            strides=1,
                            padding="same",
                            return_sequences=True,
                        )
                    )
                    self.ConvLSTM.add(
                        ConvLSTM1D(
                            num_filters,
                            kernel_size=3,
                            strides=1,
                            padding="same",
                            return_sequences=True,
                        )
                    )
                    if meta["batch_norm"]:
                        self.ConvLSTM.add(BatchNormalization())

                    self.ConvLSTM.add(
                        MaxPool2D(pool_size=(time_pool_size, space_pool_size))
                    )

            self.ConvLSTM.add(
                ConvLSTM1D(
                    meta["num_filters"][-1],
                    kernel_size=3,
                    strides=1,
                    padding="same",
                    return_sequences=False,
                )
            )
            self.ConvLSTM.add(GlobalAveragePooling1D())
        else:
            self.ConvLSTM.add(Flatten())
        if meta["units"] is not None:
            for unit in meta["units"]:
                self.ConvLSTM.add(Dense(unit, activation=meta["fc_activation"]))
        self.ConvLSTM.add(Dense(meta["summary_dim"]))

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Performs the forward pass of the model

        Args:
            x (tf.Tensor): input data of shape (batch_size, max_time_iter, nr, num_features)

        Returns:
            out (tf.Tensor): output of shape (batch_size, summary_dim)
        """
        if self.num_convlstm_blocks is not None:
            if x.shape[1] < self.min_input_time_dim:
                raise ValueError(
                    "{} - call: input x has shape {}, but applying max_pool {} times on time dimension {} leads to output shape {}".format(
                        self.__class__.__name__,
                        x.shape,
                        self.num_convlstm_blocks,
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
                        self.num_convlstm_blocks,
                        x.shape[2],
                        (
                            x.shape[0],
                            x.shape[1] // self.min_input_time_dim,
                            x.shape[2] // self.min_input_space_dim,
                            x.shape[3],
                        ),
                    )
                )

        out = self.ConvLSTM(x)
        return out


class SPM_Network(tf.keras.Model):
    def __init__(self, meta: Type[MetaDictSetting]) -> None:
        super(SPM_Network, self).__init__()

        self.ConvLSTM = ConvLSTM_Network(build_meta_dict({}, meta["ConvLSTM"]))
        self.LSTM = LSTM_Network(build_meta_dict({}, meta["LSTM"]))
        self.FC = FC_Network(build_meta_dict({}, meta["FC"]))

    def call(self, x: tf.Tensor) -> tf.Tensor:
        cs_out = self.ConvLSTM(x[:, :, :-1, :])
        v_out = self.LSTM(x[:, :, -1, :])
        feat = tf.concat([cs_out, v_out], axis=1)
        out = self.FC(feat)
        return out


class DoubleLSTM_Network(tf.keras.Model):
    def __init__(self, meta: Type[MetaDictSetting]) -> None:
        super(DoubleLSTM_Network, self).__init__()

        self.LSTM_time = LSTM_Network(build_meta_dict({}, meta["LSTM"]))
        self.LSTM_space = LSTM_Network(build_meta_dict({}, meta["LSTM"]))
        self.FC = FC_Network(build_meta_dict({}, meta["FC"]))

    def call(self, x: tf.Tensor) -> tf.Tensor:
        time_feat = self.LSTM_time(x)
        space_dim = x.shape[2] // 2
        u_trans = K.permute_dimensions(x[:, :, :space_dim], (0, 2, 1))
        v_trans = K.permute_dimensions(x[:, :, space_dim:], (0, 2, 1))
        x_trans = tf.concat([u_trans, v_trans], axis=1)
        space_feat = self.LSTM_space(x_trans)
        out = self.FC(tf.concat([time_feat, space_feat], axis=1))
        return out
