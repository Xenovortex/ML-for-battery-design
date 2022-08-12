import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential


class FC_net(tf.keras.Model):
    """Implements a fully-connected network with keras.

    Attributes:
        FC (tf.keras.Model): fully-connected network architecture
    """

    def __init__(self, meta) -> None:
        super(FC_net, self).__init__()

        self.FC = Sequential(
            [Flatten()]
            + [Dense(unit, activation=meta["activation"]) for unit in meta["units"]]
            + [Dense(meta["summary_dim"], activation="sigmoid")]
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Performs the forward pass of the model

        Args:
            x (tf.Tensor): input data of shape (batch_size, max_time_iter, (nr), num_features)

        Returns:
            out (tf.Tensor): output of shape (batch_size, summary_dim)
        """
        out = self.FC(x)
        return out
