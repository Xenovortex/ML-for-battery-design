import io
import random
import sys

import pytest
import tensorflow as tf
from bayesflow.default_settings import MetaDictSetting
from bayesflow.helper_functions import build_meta_dict
from parameterized import parameterized
from tensorflow.keras.activations import elu, relu, sigmoid, tanh
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

    @parameterized.expand([[0], [random.randint(1, 3)]])
    def test_ode_inference(self, num_layer):
        summary_dim = random.randint(1, 4)
        units = (
            [random.randint(4, 16) for _ in range(num_layer)]
            if (num_layer > 0)
            else None
        )

        architecture_settings = MetaDictSetting(
            meta_dict={
                "units": units,
                "activation": "relu",
                "summary_dim": summary_dim,
            }
        )
        model = FC_Network(build_meta_dict({}, architecture_settings))

        batch_size = random.randint(1, 8)
        max_time_iter = random.randint(1, 10)
        num_features = random.randint(1, 10)
        input_data = tf.random.uniform((batch_size, max_time_iter, num_features))
        output = model(input_data)
        if units is not None:
            expected_layer_types = [Flatten] + num_layer * [Dense] + [Dense]
            expected_activation = [None] + num_layer * [relu] + [sigmoid]
            expected_layer_output_shape = (
                [(batch_size, max_time_iter * num_features)]
                + [(batch_size, unit) for unit in units]
                + [(batch_size, summary_dim)]
            )
        else:
            expected_layer_types = [Flatten, Dense]
            expected_activation = [None, sigmoid]
            expected_layer_output_shape = [
                (batch_size, max_time_iter * num_features)
            ] + [(batch_size, summary_dim)]

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
        self.assertEqual(output.shape[1], summary_dim)
        self.assertEqual(len(model.FC.layers), num_layer + 2)
        for layer, layer_type, layer_act, layer_out_shape in zip(
            model.FC.layers,
            expected_layer_types,
            expected_activation,
            expected_layer_output_shape,
        ):
            assert isinstance(layer, layer_type)
            if layer_act is not None:
                self.assertEqual(layer.activation, layer_act)
            self.assertEqual(layer.output_shape, layer_out_shape)

    @parameterized.expand([[0], [random.randint(1, 3)]])
    def test_pde_inference(self, num_layer):
        summary_dim = random.randint(1, 4)
        if num_layer == 0:
            units = None
        else:
            units = [random.randint(4, 16) for _ in range(num_layer)]

        architecture_settings = MetaDictSetting(
            meta_dict={
                "units": units,
                "activation": "relu",
                "summary_dim": summary_dim,
            }
        )
        model = FC_Network(build_meta_dict({}, architecture_settings))

        batch_size = random.randint(1, 8)
        max_time_iter = random.randint(1, 10)
        nr = random.randint(1, 10)
        num_features = random.randint(1, 10)
        input_data = tf.random.uniform((batch_size, max_time_iter, nr, num_features))
        output = model(input_data)
        if units is not None:
            expected_layer_types = [Flatten] + num_layer * [Dense] + [Dense]
            expected_activation = [None] + num_layer * [relu] + [sigmoid]
            expected_layer_output_shape = (
                [(batch_size, max_time_iter * nr * num_features)]
                + [(batch_size, unit) for unit in units]
                + [(batch_size, summary_dim)]
            )
        else:
            expected_layer_types = [Flatten, Dense]
            expected_activation = [None, sigmoid]
            expected_layer_output_shape = [
                (batch_size, max_time_iter * nr * num_features)
            ] + [(batch_size, summary_dim)]

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
        self.assertEqual(output.shape[1], summary_dim)
        self.assertEqual(len(model.FC.layers), num_layer + 2)
        for layer, layer_type, layer_act, layer_out_shape in zip(
            model.FC.layers,
            expected_layer_types,
            expected_activation,
            expected_layer_output_shape,
        ):
            assert isinstance(layer, layer_type)
            if layer_act is not None:
                self.assertEqual(layer.activation, layer_act)
            self.assertEqual(layer.output_shape, layer_out_shape)


class LSTMTest(tf.test.TestCase):
    def setUp(self):
        super(LSTMTest, self).setUp()

    @parameterized.expand(
        [
            [0, 0],
            [0, random.randint(1, 3)],
            [random.randint(1, 3), 0],
            [random.randint(1, 3), random.randint(1, 3)],
        ]
    )
    def test_ode_inference(self, lstm_num_layer, fc_num_layer):
        summary_dim = random.randint(1, 4)
        fc_units = (
            [random.randint(4, 16) for _ in range(fc_num_layer)]
            if (fc_num_layer > 0)
            else None
        )
        lstm_units = (
            [random.randint(4, 16) for _ in range(lstm_num_layer)]
            if (lstm_num_layer > 0)
            else None
        )
        architecture_settings = MetaDictSetting(
            meta_dict={
                "lstm_units": lstm_units,
                "fc_units": fc_units,
                "fc_activation": "relu",
                "summary_dim": summary_dim,
            }
        )
        model = LSTM_Network(build_meta_dict({}, architecture_settings))

        batch_size = random.randint(1, 8)
        max_time_iter = random.randint(1, 10)
        num_features = random.randint(1, 10)
        input_data = tf.random.uniform((batch_size, max_time_iter, num_features))
        output = model(input_data)
        expected_layer_types = []
        expected_activation = []
        expected_layer_output_shape = []
        if lstm_units is not None:
            expected_layer_types += lstm_num_layer * [LSTM]
            expected_activation += lstm_num_layer * [(tanh, sigmoid)]
            if lstm_num_layer > 1:
                expected_layer_output_shape += [
                    (batch_size, max_time_iter, unit) for unit in lstm_units[:-1]
                ]
            expected_layer_output_shape += [(batch_size, lstm_units[-1])]
            num_flatten_layer = 0
        else:
            expected_layer_types += [Flatten]
            expected_activation += [None]
            expected_layer_output_shape += [(batch_size, max_time_iter * num_features)]
            num_flatten_layer = 1
        if fc_units is not None:
            expected_layer_types += fc_num_layer * [Dense]
            expected_activation += fc_num_layer * [relu]
            expected_layer_output_shape += [(batch_size, unit) for unit in fc_units]
        expected_layer_types += [Dense]
        expected_activation += [sigmoid]
        expected_layer_output_shape += [(batch_size, summary_dim)]

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
        self.assertEqual(output.shape[1], summary_dim)
        self.assertEqual(
            len(model.LSTM.layers),
            lstm_num_layer + fc_num_layer + num_flatten_layer + 1,
        )
        for layer, layer_type, layer_act, layer_out_shape in zip(
            model.LSTM.layers,
            expected_layer_types,
            expected_activation,
            expected_layer_output_shape,
        ):
            assert isinstance(layer, layer_type)
            if isinstance(layer, LSTM):
                self.assertEqual(layer.activation, layer_act[0])
                self.assertEqual(layer.recurrent_activation, layer_act[1])
            if isinstance(layer, Dense):
                self.assertEqual(layer.activation, layer_act)
            self.assertEqual(layer.output_shape, layer_out_shape)


class CNNTest(tf.test.TestCase):
    def setUp(self):
        super(CNNTest, self).setUp()

    @parameterized.expand(
        [
            [True, True, random.randint(1, 3), random.randint(1, 3)],
            [True, False, random.randint(1, 3), random.randint(1, 3)],
            [False, True, random.randint(1, 3), random.randint(1, 3)],
            [False, False, random.randint(1, 3), random.randint(1, 3)],
            [
                random.choice([True, False]),
                random.choice([True, False]),
                0,
                random.randint(1, 3),
            ],
            [
                random.choice([True, False]),
                random.choice([True, False]),
                random.randint(1, 3),
                0,
            ],
            [random.choice([True, False]), random.choice([True, False]), 0, 0],
        ]
    )
    def test_pde_inference(self, pool_time, pool_space, cnn_num_blocks, fc_num_layer):
        summary_dim = random.randint(1, 4)
        num_filters = (
            [random.randint(1, 16) for _ in range(cnn_num_blocks)]
            if (cnn_num_blocks > 0)
            else None
        )
        units = (
            [random.randint(4, 16) for _ in range(fc_num_layer)]
            if (fc_num_layer > 0)
            else None
        )

        architecture_settings = MetaDictSetting(
            meta_dict={
                "num_filters": num_filters,
                "cnn_activation": "elu",
                "units": units,
                "fc_activation": "relu",
                "summary_dim": summary_dim,
                "pool_time": pool_time,
                "pool_space": pool_space,
            }
        )

        model = CNN_Network(build_meta_dict({}, architecture_settings))

        batch_size = random.randint(1, 32)
        max_time_iter = 2 ** random.randint(cnn_num_blocks, 6)
        nr = 2 ** random.randint(cnn_num_blocks, 6)
        num_features = random.randint(1, 10)
        input_data = tf.random.uniform((batch_size, max_time_iter, nr, num_features))
        output = model(input_data)
        expected_layer_types = []
        expected_activation = []
        expected_layer_output_shape = []
        if num_filters is not None:
            expected_layer_types += cnn_num_blocks * [Conv2D, Conv2D, MaxPool2D] + [
                GlobalAveragePooling2D
            ]
            expected_activation += cnn_num_blocks * [elu, elu, None] + [None]
            temp_max_time_iter = max_time_iter
            temp_nr = nr
            for num_filter in num_filters:
                new_max_time_iter = (
                    temp_max_time_iter // 2 if pool_time else temp_max_time_iter
                )
                new_nr = temp_nr // 2 if pool_space else temp_nr
                expected_layer_output_shape += 2 * [
                    (batch_size, temp_max_time_iter, temp_nr, num_filter)
                ] + [(batch_size, new_max_time_iter, new_nr, num_filter)]
                temp_max_time_iter = new_max_time_iter
                temp_nr = new_nr
            expected_layer_output_shape += [(batch_size, num_filters[-1])]
        else:
            expected_layer_types += [Flatten]
            expected_activation += [None]
            expected_layer_output_shape += [
                (batch_size, max_time_iter * nr * num_features)
            ]
        if units is not None:
            expected_layer_types += fc_num_layer * [Dense]
            expected_activation += fc_num_layer * [relu]
            expected_layer_output_shape += [(batch_size, unit) for unit in units]
        expected_layer_types += [Dense]
        expected_activation += [sigmoid]
        expected_layer_output_shape += [(batch_size, summary_dim)]

        assert isinstance(input_data, tf.Tensor)
        self.assertEqual(input_data.dtype, tf.dtypes.float32)
        self.assertEqual(len(input_data.shape), 4)
        self.assertEqual(input_data.shape[0], batch_size)
        self.assertEqual(input_data.shape[1], max_time_iter)
        self.assertEqual(input_data.shape[2], nr)
        self.assertEqual(input_data.shape[3], num_features)
        if num_filters is not None:
            self.assertEqual(cnn_num_blocks, len(num_filters))
        else:
            self.assertEqual(cnn_num_blocks, 0)
        if units is not None:
            self.assertEqual(fc_num_layer, len(units))
        else:
            self.assertEqual(fc_num_layer, 0)
        assert isinstance(pool_time, bool)
        assert isinstance(pool_space, bool)
        self.assertEqual(model.num_cnn_blocks, cnn_num_blocks)
        if pool_time:
            self.assertEqual(model.min_input_time_dim, 2**cnn_num_blocks)
        else:
            self.assertEqual(model.min_input_time_dim, 1)
        if pool_space:
            self.assertEqual(model.min_input_space_dim, 2**cnn_num_blocks)
        else:
            self.assertEqual(model.min_input_space_dim, 1)
        assert isinstance(output, tf.Tensor)
        self.assertEqual(output.dtype, tf.dtypes.float32)
        self.assertEqual(len(output.shape), 2)
        self.assertEqual(output.shape[0], batch_size)
        self.assertEqual(output.shape[1], summary_dim)
        self.assertEqual(len(model.CNN.layers), 3 * cnn_num_blocks + fc_num_layer + 2)
        for layer, layer_type, layer_act, layer_out_shape in zip(
            model.CNN.layers,
            expected_layer_types,
            expected_activation,
            expected_layer_output_shape,
        ):
            assert isinstance(layer, layer_type)
            if isinstance(layer, (Dense, Conv2D)):
                self.assertEqual(layer.activation, layer_act)
            self.assertEqual(layer.output_shape, layer_out_shape)

    @parameterized.expand(
        [
            [random.randint(1, 7), random.randint(32, 100), "time_error"],
            [random.randint(32, 100), random.randint(1, 7), "space_error"],
            [random.randint(1, 7), random.randint(1, 7), "time_error"],
        ]
    )
    def test_max_pool_leads_to_zero_dim(self, max_time_iter, nr, expected_error):
        summary_dim = random.randint(1, 4)
        cnn_num_blocks = random.randint(3, 5)
        fc_num_layer = random.randint(1, 3)
        num_filters = (
            [random.randint(1, 16) for _ in range(cnn_num_blocks)]
            if (cnn_num_blocks > 0)
            else None
        )
        units = (
            [random.randint(4, 16) for _ in range(fc_num_layer)]
            if (fc_num_layer > 0)
            else None
        )

        architecture_settings = MetaDictSetting(
            meta_dict={
                "num_filters": num_filters,
                "cnn_activation": "elu",
                "units": units,
                "fc_activation": "relu",
                "summary_dim": summary_dim,
                "pool_time": True,
                "pool_space": True,
            }
        )

        model = CNN_Network(build_meta_dict({}, architecture_settings))

        batch_size = random.randint(1, 10)
        num_features = random.randint(1, 10)
        input_data = tf.random.uniform([batch_size, max_time_iter, nr, num_features])
        expected_output_shape = (
            input_data.shape[0],
            input_data.shape[1] // (2**cnn_num_blocks),
            input_data.shape[2] // (2**cnn_num_blocks),
            input_data.shape[3],
        )
        out = io.StringIO()
        err = io.StringIO()
        sys.stdout = out
        sys.stderr = err
        with pytest.raises(ValueError):
            model(input_data)
            self.assertEqual(out.getvalue(), "")
            if expected_error == "time_error":
                self.assertEqual(
                    err.getvalue(),
                    "{} - call: input x has shape {}, but applying max_pool {} times on time dimesion {} leads to output shape {}".format(
                        model.__class__.__name__,
                        input_data.shape,
                        cnn_num_blocks,
                        input_data.shape[1],
                        expected_output_shape,
                    ),
                )
            elif expected_error == "space_error":
                self.assertEqual(
                    err.getvalue(),
                    "{} - call: input x has shape {}, but applying max_pool {} times on space dimesion {} leads to output shape {}".format(
                        model.__class__.__name__,
                        input_data.shape,
                        cnn_num_blocks,
                        input_data.shape[2],
                        expected_output_shape,
                    ),
                )
