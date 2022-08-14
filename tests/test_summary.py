import random

import tensorflow as tf
from bayesflow.default_settings import MetaDictSetting
from bayesflow.helper_functions import build_meta_dict
from tensorflow.keras.layers import LSTM, Dense, Flatten

from ML_for_Battery_Design.src.helpers.summary import FC_Network, LSTM_Network


class FCTest(tf.test.TestCase):
    def setUp(self):
        super(FCTest, self).setUp()
        self.summary_dim, self.unit = sorted(
            [random.randint(1, 8), random.randint(1, 8)]
        )
        self.num_layer = random.randint(1, 3)
        architecture_settings = MetaDictSetting(
            meta_dict={
                "units": [self.unit] * self.num_layer,
                "activation": "relu",
                "summary_dim": self.summary_dim,
            }
        )
        self.model = FC_Network(build_meta_dict({}, architecture_settings))

    def test_ode_inference(self):
        batch_size = random.randint(1, 8)
        max_time_iter = random.randint(1, 10)
        num_features = random.randint(1, 10)
        input_data = tf.random.uniform((batch_size, max_time_iter, num_features))
        output = self.model(input_data)
        expected_layer_types = [Flatten] + self.num_layer * [Dense] + [Dense]
        expected_layer_output_shape = (
            [(batch_size, max_time_iter * num_features)]
            + [(batch_size, self.unit)] * self.num_layer
            + [(batch_size, self.summary_dim)]
        )

        assert isinstance(input_data, tf.Tensor)
        self.assertEqual(input_data.dtype, tf.dtypes.float32)
        self.assertEqual(len(input_data.shape), 3)
        self.assertEqual(input_data.shape[0], batch_size)
        self.assertEqual(input_data.shape[1], max_time_iter)
        self.assertEqual(input_data.shape[2], num_features)
        assert isinstance(output, tf.Tensor)
        self.assertEqual(output.dtype, tf.dtypes.float32)
        self.assertEqual(len(output.shape), 2)
        self.assertEqual(output.shape[0], batch_size)
        self.assertEqual(output.shape[1], self.summary_dim)
        self.assertEqual(len(self.model.FC.layers), self.num_layer + 2)
        for layer, layer_type, layer_out_shape in zip(
            self.model.FC.layers, expected_layer_types, expected_layer_output_shape
        ):
            assert isinstance(layer, layer_type)
            self.assertEqual(layer.output_shape, layer_out_shape)

    def test_pde_inference(self):
        batch_size = random.randint(1, 8)
        max_time_iter = random.randint(1, 10)
        nr = random.randint(1, 10)
        num_features = random.randint(1, 10)
        input_data = tf.random.uniform((batch_size, max_time_iter, nr, num_features))
        output = self.model(input_data)
        expected_layer_types = [Flatten] + self.num_layer * [Dense] + [Dense]
        expected_layer_output_shape = (
            [(batch_size, max_time_iter * nr * num_features)]
            + [(batch_size, self.unit)] * self.num_layer
            + [(batch_size, self.summary_dim)]
        )

        assert isinstance(input_data, tf.Tensor)
        self.assertEqual(input_data.dtype, tf.dtypes.float32)
        self.assertEqual(len(input_data.shape), 4)
        self.assertEqual(input_data.shape[0], batch_size)
        self.assertEqual(input_data.shape[1], max_time_iter)
        self.assertEqual(input_data.shape[2], nr)
        self.assertEqual(input_data.shape[3], num_features)
        assert isinstance(output, tf.Tensor)
        self.assertEqual(output.dtype, tf.dtypes.float32)
        self.assertEqual(len(output.shape), 2)
        self.assertEqual(output.shape[0], batch_size)
        self.assertEqual(output.shape[1], self.summary_dim)
        self.assertEqual(len(self.model.FC.layers), self.num_layer + 2)
        for layer, layer_type, layer_out_shape in zip(
            self.model.FC.layers, expected_layer_types, expected_layer_output_shape
        ):
            assert isinstance(layer, layer_type)
            self.assertEqual(layer.output_shape, layer_out_shape)


class LSTMTest(tf.test.TestCase):
    def setUp(self):
        super(LSTMTest, self).setUp()
        self.summary_dim, self.fc_unit, self.lstm_unit = sorted(
            [random.randint(1, 8), random.randint(1, 8), random.randint(1, 8)]
        )
        self.lstm_num_layer = random.randint(1, 3)
        self.fc_num_layer = random.randint(1, 3)
        architecture_settings = MetaDictSetting(
            meta_dict={
                "lstm_units": [self.lstm_unit] * self.lstm_num_layer,
                "fc_units": [self.fc_unit] * self.fc_num_layer,
                "fc_activation": "relu",
                "summary_dim": self.summary_dim,
            }
        )
        self.model = LSTM_Network(build_meta_dict({}, architecture_settings))

    def test_ode_inference(self):
        batch_size = random.randint(1, 8)
        max_time_iter = random.randint(1, 10)
        num_features = random.randint(1, 10)
        input_data = tf.random.uniform((batch_size, max_time_iter, num_features))
        output = self.model(input_data)
        expected_layer_types = (
            self.lstm_num_layer * [LSTM] + self.fc_num_layer * [Dense] + [Dense]
        )
        expected_layer_output_shape = (
            [(batch_size, self.lstm_unit)] * self.lstm_num_layer
            + [(batch_size, self.fc_unit)] * self.fc_num_layer
            + [(batch_size, self.summary_dim)]
        )

        assert isinstance(input_data, tf.Tensor)
        self.assertEqual(input_data.dtype, tf.dtypes.float32)
        self.assertEqual(len(input_data.shape), 3)
        self.assertEqual(input_data.shape[0], batch_size)
        self.assertEqual(input_data.shape[1], max_time_iter)
        self.assertEqual(input_data.shape[2], num_features)
        assert isinstance(output, tf.Tensor)
        self.assertEqual(output.dtype, tf.dtypes.float32)
        self.assertEqual(len(output.shape), 2)
        self.assertEqual(output.shape[0], batch_size)
        self.assertEqual(output.shape[1], self.summary_dim)
        self.assertEqual(
            len(self.model.LSTM.layers), self.lstm_num_layer + self.fc_num_layer + 1
        )
        for layer, layer_type, layer_out_shape in zip(
            self.model.LSTM.layers, expected_layer_types, expected_layer_output_shape
        ):
            assert isinstance(layer, layer_type)
            self.assertEqual(layer.output_shape, layer_out_shape)
