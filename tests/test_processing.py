import random
import string

import numpy as np
import pytest

from ML_for_Battery_Design.src.helpers.constants import (
    inference_settings,
    sim_model_collection,
    simulation_settings,
)
from ML_for_Battery_Design.src.helpers.processing import Processing

models = ["linear_ode_system"]

non_dict_input = [
    random.randint(-1000, 1000),  # int
    random.uniform(-1000, 1000),  # float
    random.uniform(-1000, 1000) + random.uniform(-1000, 1000) * 1j,  # complex
    random.choice([True, False]),  # bool
    tuple(random.random() for x in range(random.randrange(10))),  # tuple
    list(random.random() for x in range(random.randrange(10))),  # list
    set(random.random() for x in range(random.randrange(10))),  # set
    frozenset(random.random() for x in range(random.randrange(10))),  # frozenset
    "".join(random.choices(string.ascii_letters, k=random.randrange(10))),  # string,
    None,  # NoneType
]


@pytest.mark.parametrize("model_name", models)
def test_processing_init_ode(model_name):
    batch_size = random.randint(1, 8)
    max_time_iter = random.randint(1, 10)
    num_features = random.randint(1, 10)
    processing_settings = inference_settings[model_name]["processing"]
    dummy_prior_means = np.random.uniform(
        -1000, 1000, size=(batch_size, max_time_iter, num_features)
    )
    dummy_prior_stds = np.random.uniform(
        -1000, 1000, size=(batch_size, max_time_iter, num_features)
    )
    dummy_sim_data_means = np.random.uniform(
        -1000, 1000, size=(batch_size, max_time_iter, num_features)
    )
    dummy_sim_data_stds = np.random.uniform(
        -1000, 1000, size=(batch_size, max_time_iter, num_features)
    )

    configurator = Processing(
        processing_settings,
        dummy_prior_means,
        dummy_prior_stds,
        dummy_sim_data_means,
        dummy_sim_data_stds,
    )

    assert isinstance(configurator.settings, dict)
    assert isinstance(configurator.prior_means, np.ndarray)
    assert isinstance(configurator.prior_stds, np.ndarray)
    assert isinstance(configurator.sim_data_means, np.ndarray)
    assert isinstance(configurator.sim_data_stds, np.ndarray)
    assert configurator.settings == processing_settings
    assert np.array_equal(configurator.prior_means, dummy_prior_means)
    assert np.array_equal(configurator.prior_stds, dummy_prior_stds)
    assert np.array_equal(configurator.sim_data_means, dummy_sim_data_means)
    assert np.array_equal(configurator.sim_data_stds, dummy_sim_data_stds)


@pytest.mark.parametrize("model_name", models)
def test_processing_init_pde(model_name):
    batch_size = random.randint(1, 8)
    max_time_iter = random.randint(1, 10)
    nr = random.randint(1, 10)
    num_features = random.randint(1, 10)
    processing_settings = inference_settings[model_name]["processing"]
    dummy_prior_means = np.random.uniform(
        -1000, 1000, size=(batch_size, max_time_iter, nr, num_features)
    )
    dummy_prior_stds = np.random.uniform(
        -1000, 1000, size=(batch_size, max_time_iter, nr, num_features)
    )
    dummy_sim_data_means = np.random.uniform(
        -1000, 1000, size=(batch_size, max_time_iter, nr, num_features)
    )
    dummy_sim_data_stds = np.random.uniform(
        -1000, 1000, size=(batch_size, max_time_iter, nr, num_features)
    )

    configurator = Processing(
        processing_settings,
        dummy_prior_means,
        dummy_prior_stds,
        dummy_sim_data_means,
        dummy_sim_data_stds,
    )

    assert isinstance(configurator.settings, dict)
    assert isinstance(configurator.prior_means, np.ndarray)
    assert isinstance(configurator.prior_stds, np.ndarray)
    assert isinstance(configurator.sim_data_means, np.ndarray)
    assert isinstance(configurator.sim_data_stds, np.ndarray)
    assert configurator.settings == processing_settings
    assert np.array_equal(configurator.prior_means, dummy_prior_means)
    assert np.array_equal(configurator.prior_stds, dummy_prior_stds)
    assert np.array_equal(configurator.sim_data_means, dummy_sim_data_means)
    assert np.array_equal(configurator.sim_data_stds, dummy_sim_data_stds)


@pytest.mark.parametrize(
    "test_cases",
    [
        [(None, None, None, None), ("norm_prior is True", "prior_means")],
        [
            (
                None,
                None,
                np.random.uniform(
                    -1000,
                    1000,
                    size=(
                        random.randint(1, 8),
                        random.randint(1, 10),
                        random.randint(1, 10),
                    ),
                ),
                np.random.uniform(
                    -1000,
                    1000,
                    size=(
                        random.randint(1, 8),
                        random.randint(1, 10),
                        random.randint(1, 10),
                    ),
                ),
            ),
            ("norm_prior is True", "prior_means"),
        ],
        [
            (
                np.random.uniform(
                    -1000,
                    1000,
                    size=(
                        random.randint(1, 8),
                        random.randint(1, 10),
                        random.randint(1, 10),
                    ),
                ),
                np.random.uniform(
                    -1000,
                    1000,
                    size=(
                        random.randint(1, 8),
                        random.randint(1, 10),
                        random.randint(1, 10),
                    ),
                ),
                None,
                None,
            ),
            ("norm_sim_data is mean_std", "sim_data_means"),
        ],
        [
            (
                None,
                np.random.uniform(
                    -1000,
                    1000,
                    size=(
                        random.randint(1, 8),
                        random.randint(1, 10),
                        random.randint(1, 10),
                    ),
                ),
                np.random.uniform(
                    -1000,
                    1000,
                    size=(
                        random.randint(1, 8),
                        random.randint(1, 10),
                        random.randint(1, 10),
                    ),
                ),
                np.random.uniform(
                    -1000,
                    1000,
                    size=(
                        random.randint(1, 8),
                        random.randint(1, 10),
                        random.randint(1, 10),
                    ),
                ),
            ),
            ("norm_prior is True", "prior_means"),
        ],
        [
            (
                np.random.uniform(
                    -1000,
                    1000,
                    size=(
                        random.randint(1, 8),
                        random.randint(1, 10),
                        random.randint(1, 10),
                    ),
                ),
                None,
                np.random.uniform(
                    -1000,
                    1000,
                    size=(
                        random.randint(1, 8),
                        random.randint(1, 10),
                        random.randint(1, 10),
                    ),
                ),
                np.random.uniform(
                    -1000,
                    1000,
                    size=(
                        random.randint(1, 8),
                        random.randint(1, 10),
                        random.randint(1, 10),
                    ),
                ),
            ),
            ("norm_prior is True", "prior_stds"),
        ],
        [
            (
                np.random.uniform(
                    -1000,
                    1000,
                    size=(
                        random.randint(1, 8),
                        random.randint(1, 10),
                        random.randint(1, 10),
                    ),
                ),
                np.random.uniform(
                    -1000,
                    1000,
                    size=(
                        random.randint(1, 8),
                        random.randint(1, 10),
                        random.randint(1, 10),
                    ),
                ),
                None,
                np.random.uniform(
                    -1000,
                    1000,
                    size=(
                        random.randint(1, 8),
                        random.randint(1, 10),
                        random.randint(1, 10),
                    ),
                ),
            ),
            ("norm_sim_data is mean_std", "sim_data_means"),
        ],
        [
            (
                np.random.uniform(
                    -1000,
                    1000,
                    size=(
                        random.randint(1, 8),
                        random.randint(1, 10),
                        random.randint(1, 10),
                    ),
                ),
                np.random.uniform(
                    -1000,
                    1000,
                    size=(
                        random.randint(1, 8),
                        random.randint(1, 10),
                        random.randint(1, 10),
                    ),
                ),
                np.random.uniform(
                    -1000,
                    1000,
                    size=(
                        random.randint(1, 8),
                        random.randint(1, 10),
                        random.randint(1, 10),
                    ),
                ),
                None,
            ),
            ("norm_sim_data is mean_std", "sim_data_stds"),
        ],
    ],
)
def test_processing_init_value_errors(test_cases, capsys):
    args, expected_error = test_cases
    processing_settings = {
        "norm_prior": True,
        "norm_sim_data": "mean_std",
        "remove_nan": True,
    }

    with pytest.raises(ValueError):
        configurator = Processing(processing_settings, *args)
        out, err = capsys.readouterr()
        assert out == ""
        assert err == "{} - init: processing setting {}, but {} is None".format(
            configurator.__class__.__name__, expected_error[0], expected_error[1]
        )


@pytest.mark.parametrize("non_dict_input", non_dict_input)
def test_processing_init_type_error(non_dict_input, capsys):

    with pytest.raises(TypeError):
        Processing(non_dict_input)
        out, err = capsys.readouterr()
        assert out == ""
        assert err == "{} - init: processing_settings input is not dictionary type"


@pytest.mark.parametrize("model_name", models)
def test_processing_call_norm_prior(model_name):
    processing_settings = {
        "norm_prior": True,
        "norm_sim_data": None,
        "remove_nan": False,
    }

    test_object = sim_model_collection[model_name](**simulation_settings[model_name])
    batch_size = random.randint(1, 8)
    data_dict = test_object.generative_model(batch_size=batch_size)
    means = np.mean(data_dict["prior_draws"], axis=0, keepdims=True)
    stds = np.std(data_dict["prior_draws"], axis=0, keepdims=True)

    configurator = Processing(processing_settings, prior_means=means, prior_stds=stds)

    out_dict = configurator(data_dict)

    assert isinstance(configurator.settings, dict)
    assert isinstance(configurator.prior_means, np.ndarray)
    assert isinstance(configurator.prior_stds, np.ndarray)
    assert configurator.settings == processing_settings
    assert np.array_equal(configurator.prior_means, means)
    assert np.array_equal(configurator.prior_stds, stds)
    assert configurator.sim_data_means is None
    assert configurator.sim_data_stds is None
    assert out_dict["parameters"].ndim == 2
    assert out_dict["parameters"].shape[0] == batch_size
    assert out_dict["parameters"].shape[1] == test_object.num_hidden_params
    assert out_dict["summary_conditions"].shape[0] == batch_size
    assert out_dict["summary_conditions"].shape[1] == test_object.max_time_iter
    if test_object.is_pde:
        assert out_dict["summary_conditions"].ndim == 4
        assert out_dict["summary_conditions"].shape[2] == test_object.nr
        assert out_dict["summary_conditions"].shape[3] == test_object.num_features
    else:
        assert out_dict["summary_conditions"].ndim == 3
        assert out_dict["summary_conditions"].shape[2] == test_object.num_features
    assert np.allclose(
        np.nan_to_num(np.mean(out_dict["parameters"], axis=0), nan=0),
        np.zeros(test_object.num_hidden_params),
        atol=1e-6,
    )
    assert np.allclose(
        np.nan_to_num(np.std(out_dict["parameters"], axis=0), nan=1),
        np.ones(test_object.num_hidden_params),
        atol=1e-6,
    )
    assert np.array_equal(out_dict["summary_conditions"], data_dict["sim_data"])


@pytest.mark.parametrize("model_name", models)
def test_processing_call_norm_sim_data_log(model_name):
    processing_settings = {
        "norm_prior": False,
        "norm_sim_data": "log_norm",
        "remove_nan": False,
    }

    test_object = sim_model_collection[model_name](**simulation_settings[model_name])
    batch_size = random.randint(1, 8)
    data_dict = test_object.generative_model(batch_size=batch_size)

    configurator = Processing(processing_settings)

    out_dict = configurator(data_dict)

    assert isinstance(configurator.settings, dict)
    assert configurator.settings == processing_settings
    assert configurator.prior_means is None
    assert configurator.prior_stds is None
    assert configurator.sim_data_means is None
    assert configurator.sim_data_stds is None
    assert out_dict["parameters"].ndim == 2
    assert out_dict["parameters"].shape[0] == batch_size
    assert out_dict["parameters"].shape[1] == test_object.num_hidden_params
    assert out_dict["summary_conditions"].shape[0] == batch_size
    assert out_dict["summary_conditions"].shape[1] == test_object.max_time_iter
    if test_object.is_pde:
        assert out_dict["summary_conditions"].ndim == 4
        assert out_dict["summary_conditions"].shape[2] == test_object.nr
        assert out_dict["summary_conditions"].shape[3] == test_object.num_features
    else:
        assert out_dict["summary_conditions"].ndim == 3
        assert out_dict["summary_conditions"].shape[2] == test_object.num_features
    assert np.array_equal(out_dict["parameters"], data_dict["prior_draws"])
    assert np.array_equal(
        out_dict["summary_conditions"],
        np.log1p(data_dict["sim_data"]),
        equal_nan=True,
    )


@pytest.mark.parametrize("model_name", models)
def test_processing_call_norm_sim_data_mean_std(model_name):
    processing_settings = {
        "norm_prior": False,
        "norm_sim_data": "mean_std",
        "remove_nan": False,
    }

    test_object = sim_model_collection[model_name](**simulation_settings[model_name])
    batch_size = random.randint(1, 8)
    data_dict = test_object.generative_model(batch_size=batch_size)
    means = np.mean(data_dict["sim_data"], axis=0, keepdims=True)
    stds = np.std(data_dict["sim_data"], axis=0, keepdims=True)

    configurator = Processing(
        processing_settings, sim_data_means=means, sim_data_stds=stds
    )

    out_dict = configurator(data_dict)

    assert isinstance(configurator.settings, dict)
    assert isinstance(configurator.sim_data_means, np.ndarray)
    assert isinstance(configurator.sim_data_stds, np.ndarray)
    assert configurator.settings == processing_settings
    assert configurator.prior_means is None
    assert configurator.prior_stds is None
    assert np.array_equal(configurator.sim_data_means, means)
    assert np.array_equal(configurator.sim_data_stds, stds)
    assert out_dict["parameters"].ndim == 2
    assert out_dict["parameters"].shape[0] == batch_size
    assert out_dict["parameters"].shape[1] == test_object.num_hidden_params
    assert out_dict["summary_conditions"].shape[0] == batch_size
    assert out_dict["summary_conditions"].shape[1] == test_object.max_time_iter
    if test_object.is_pde:
        assert out_dict["summary_conditions"].ndim == 4
        assert out_dict["summary_conditions"].shape[2] == test_object.nr
        assert out_dict["summary_conditions"].shape[3] == test_object.num_features
    else:
        assert out_dict["summary_conditions"].ndim == 3
        assert out_dict["summary_conditions"].shape[2] == test_object.num_features
    assert np.array_equal(out_dict["parameters"], data_dict["prior_draws"])
    if test_object.is_pde:
        assert np.allclose(
            np.nan_to_num(np.mean(out_dict["summary_conditions"], axis=0), nan=0),
            np.zeros(
                (test_object.max_time_iter, test_object.nr, test_object.num_features)
            ),
            atol=1e-5,
        )
        assert np.allclose(
            np.nan_to_num(np.std(out_dict["summary_conditions"], axis=0), nan=1),
            np.ones(
                (test_object.max_time_iter, test_object.nr, test_object.num_features)
            ),
            atol=1e-5,
        )
    else:
        assert np.allclose(
            np.nan_to_num(np.mean(out_dict["summary_conditions"], axis=0), nan=0),
            np.zeros((test_object.max_time_iter, test_object.num_features)),
            atol=1e-5,
        )
        assert np.allclose(
            np.nan_to_num(np.std(out_dict["summary_conditions"], axis=0), nan=1),
            np.ones((test_object.max_time_iter, test_object.num_features)),
            atol=1e-5,
        )


@pytest.mark.parametrize("model_name", models)
def test_processing_call_remove_nan(model_name):
    processing_settings = {
        "norm_prior": False,
        "norm_sim_data": None,
        "remove_nan": True,
    }

    test_object = sim_model_collection[model_name](**simulation_settings[model_name])
    batch_size = random.randint(1, 8)
    data_dict = test_object.generative_model(batch_size=batch_size)
    nan_mask = np.random.choice(
        [True, False], size=data_dict["sim_data"].shape, p=(0.1, 0.9)
    )
    data_dict["sim_data"][nan_mask] = np.nan

    configurator = Processing(processing_settings)

    out_dict = configurator(data_dict)

    assert isinstance(configurator.settings, dict)
    assert configurator.settings == processing_settings
    assert configurator.prior_means is None
    assert configurator.prior_stds is None
    assert configurator.sim_data_means is None
    assert configurator.sim_data_stds is None
    assert out_dict["parameters"].shape[0] == out_dict["summary_conditions"].shape[0]
    assert out_dict["parameters"].ndim == 2
    assert out_dict["parameters"].shape[1] == test_object.num_hidden_params
    assert out_dict["summary_conditions"].shape[1] == test_object.max_time_iter
    if test_object.is_pde:
        assert out_dict["summary_conditions"].ndim == 4
        assert out_dict["summary_conditions"].shape[2] == test_object.nr
        assert out_dict["summary_conditions"].shape[3] == test_object.num_features
    else:
        assert out_dict["summary_conditions"].ndim == 3
        assert out_dict["summary_conditions"].shape[2] == test_object.num_features
    assert not np.any(np.isnan(out_dict["summary_conditions"]))


@pytest.mark.parametrize("model_name", models)
def test_procesing_call_value_error(model_name, capsys):
    processing_settings = {
        "norm_prior": False,
        "norm_sim_data": "".join(
            random.choices(string.ascii_letters, k=random.randrange(10))
        ),
        "remove_nan": False,
    }

    test_object = sim_model_collection[model_name](**simulation_settings[model_name])
    batch_size = random.randint(1, 8)
    data_dict = test_object.generative_model(batch_size=batch_size)

    configurator = Processing(processing_settings)

    with pytest.raises(ValueError):
        configurator(data_dict)
        out, err = capsys.readouterr()
        assert out == ""
        assert (
            err
            == "{} - call: processing setting norm_sim_data '{}' is not a valid input".format(
                configurator.__class__.__name__, processing_settings["norm_sim_data"]
            )
        )
