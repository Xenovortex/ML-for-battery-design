import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


class SummaryNetArmortizer(tf.keras.Model):
    def __init__(self, summary_net, normalize=None, **kwargs) -> None:
        super(SummaryNetArmortizer, self).__init__(**kwargs)

        self.summary_net = summary_net
        self.normalize = normalize

    def call(self, input_dict, **kwargs):
        out = self.summary_net(input_dict["summary_conditions"])
        return out

    def compute_loss(self, input_dict, **kwargs):
        true_params = K.constant(input_dict["parameters"])
        sim_data = K.constant(input_dict["summary_conditions"])
        pred_params = self.summary_net(sim_data)
        rmse = K.sqrt(K.mean(K.square(true_params - pred_params), axis=0))

        if self.normalize is None:
            loss = K.sum(rmse)
        elif self.normalize == "std":
            loss = K.sum(rmse / K.std(true_params, axis=0))
        elif self.normalize == "mean":
            loss = K.sum(rmse / K.mean(true_params, axis=0))
        elif self.normalize == "minmax":
            loss = K.sum(
                rmse / (np.max(true_params, axis=0) - np.min(true_params, axis=0))
            )
        elif self.normalize == "iqr":
            loss = K.sum(
                rmse
                / (
                    np.quantile(true_params, 0.75, axis=0)
                    - np.quantile(true_params, 0.25, axis=0)
                )
            )
        else:
            raise ValueError("normalize must be 'std', 'mean', 'minmax' or 'iqr'")

        return loss

    def sample(self, input_dict, n_samples, to_numpy=True, **kwargs):
        out = self.summary_net(input_dict["summary_conditions"])
        post_samples = tf.repeat(tf.expand_dims(out, axis=1), repeats=n_samples, axis=1)
        if to_numpy:
            return post_samples.numpy()
        return post_samples
