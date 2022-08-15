import random

import tensorflow as tf
from bayesflow.default_settings import MetaDictSetting
from bayesflow.helper_functions import build_meta_dict
from tensorflow.keras.layers import (
    LSTM,
    Conv2D,
    Dense,
    Flatten,
    GlobalAveragePooling2D,
    MaxPool2D,
)

from ML_for_Battery_Design.src.helpers.summary import (
    CNN_Network,
    FC_Network,
    LSTM_Network,
)


class FCTest(tf.test.TestCase):
    def setUp(self):
        super(FCTest, self).setUp()
        self.summary_dim = random.randint(1, 4)
        self.num_layer = random.randint(1, 3)
        self.units = [random.randint(4, 16) for _ in range(self.num_layer)]

        architecture_settings = MetaDictSetting(
            meta_dict={
                "units": self.units,
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
            + [(batch_size, unit) for unit in self.units]
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
            + [(batch_size, unit) for unit in self.units]
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
        self.summary_dim = random.randint(1, 4)
        self.lstm_num_layer = random.randint(1, 3)
        self.fc_num_layer = random.randint(1, 3)
        self.fc_units = [random.randint(4, 16) for _ in range(self.fc_num_layer)]
        self.lstm_units = [random.randint(4, 16) for _ in range(self.lstm_num_layer)]
        architecture_settings = MetaDictSetting(
            meta_dict={
                "lstm_units": self.lstm_units,
                "fc_units": self.fc_units,
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
            [(batch_size, unit) for unit in self.lstm_units]
            + [(batch_size, unit) for unit in self.fc_units]
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


class CNNTest(tf.test.TestCase):
    def setUp(self):
        super(CNNTest, self).setUp()

    @staticmethod
    def subtest_pde_inference(self, pool_time, pool_space):
        summary_dim = random.randint(1, 4)
        cnn_num_blocks = random.randint(1, 3)
        num_filters = [random.randint(1, 16) for _ in range(cnn_num_blocks)]
        kernel_size = [random.randint(1, 5) for _ in range(cnn_num_blocks)]
        stride = [random.randint(1, 3) for _ in range(cnn_num_blocks)]
        fc_num_layer = random.randint(1, 3)
        units = [random.randint(4, 16) for _ in range(fc_num_layer)]

        architecture_settings = MetaDictSetting(
            meta_dict={
                "num_filters": num_filters,
                "kernel_size": kernel_size,
                "stride": stride,
                "cnn_activation": "elu",
                "units": units,
                "fc_activation": "relu",
                "summary_dim": summary_dim,
                "pool_time": pool_time,
                "pool_space": pool_space,
            }
        )

        model = CNN_Network(build_meta_dict({}, architecture_settings))

        batch_size = random.randint(1, 8)
        max_time_iter = random.randint(1, 10)
        nr = random.randint(1, 10)
        num_features = random.randint(1, 10)
        input_data = tf.random.uniform((batch_size, max_time_iter, nr, num_features))
        output = model(input_data)
        expected_layer_types = (
            [Conv2D, Conv2D, MaxPool2D] * cnn_num_blocks
            + [GlobalAveragePooling2D]
            + [Dense] * fc_num_layer
            + [Dense]
        )
        expected_layer_output_shape = []
        for i, num_filter in enumerate(num_filters):
            new_max_time_iter = max_time_iter // 2 if pool_time else max_time_iter
            new_nr = nr // 2 if pool_space else nr
            expected_layer_output_shape += 2 * [
                (batch_size, max_time_iter, nr, num_filter)
            ] + [(batch_size, new_max_time_iter, new_nr, num_filter)]
        expected_layer_output_shape += [(batch_size, num_filters[-1])]
        expected_layer_output_shape += [(batch_size, unit) for unit in units]
        expected_layer_output_shape += [(batch_size, summary_dim)]

        assert isinstance(input_data, tf.Tensor)
        assert isinstance(pool_time, bool)
        assert isinstance(pool_space, bool)
        self.assertEqual(input_data.dtype, tf.dtypes.float32)
        self.assertEqual(len(input_data.shape), 4)
        self.assertEqual(input_data.shape[0], batch_size)
        self.assertEqual(input_data.shape[1], max_time_iter)
        self.assertEqual(input_data.shape[2], nr)
        self.assertEqual(input_data.shape[3], num_features)
        self.assertEqual(cnn_num_blocks, len(num_filters))
        self.assertEqual(cnn_num_blocks, len(kernel_size))
        self.assertEqual(cnn_num_blocks, len(stride))
        self.assertEqual(fc_num_layer, len(units))
        assert isinstance(output, tf.Tensor)
        self.assertEqual(output.dtype, tf.dtypes.float32)
        self.assertEqual(len(output.shape), 2)
        self.assertEqual(output.shape[0], batch_size)
        self.assertEqual(output.shape[1], self.summary_dim)
        self.assertEqual(len(self.model.CNN.layers), cnn_num_blocks + fc_num_layer + 2)
        for layer, layer_type, layer_out_shape in zip(
            self.model.CNN.layers, expected_layer_types, expected_layer_output_shape
        ):
            assert isinstance(layer, layer_type)
            self.assertEqual(layer.output_shape, layer_out_shape)
