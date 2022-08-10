import random

import numpy as np
import pytest

from ML_for_Battery_Design.src.settings.simulation.linear_ode_settings import (
    LINEAR_ODE_SYSTEM_SETTINGS,
)
from ML_for_Battery_Design.src.simulation.linear_ode_model import LinearODEsystem

# ------------------------------ Dummy Test Data ----------------------------- #

dummy_matrices = [
    (np.array([1, 2, 3, 4]), True),  # real, negative and positive
    (np.array([1, 2, 1, 4]), True),  # real, both positive
    (np.array([-1, 0, 0, -3]), False),  # real, both negative
    (np.array([0, 0, 0, 0]), False),  # real, both zero
    (np.array([-1, 0, 0, 0]), False),  # real, negative and zero
    (np.array([1, 0, 0, 0]), True),  # real, positive and zero
    (np.array([1, -2, 3, 4]), True),  # complex, both real part positive
    (np.array([1, -2, 5, -1]), False),  # complex, both real part negative
    (np.array([1, -4, 4, -1]), False),  # complex, both real part zero
]


# ---------------- Test LinearODEsystem Class Inititialization --------------- #


def test_linear_ode_system_init(capsys):
    test_object = LinearODEsystem(**LINEAR_ODE_SYSTEM_SETTINGS)
    out, err = capsys.readouterr()
    expected_output = (
        80 * "#"
        + "\n"
        + "\n"
        + "Initialize simulation model: LinearODEsystem\n"
        + 80 * "-"
        + "\n"
        + "hidden parameters: {}\n".format(test_object.hidden_param_names)
        + "dt0: {}\n".format(LINEAR_ODE_SYSTEM_SETTINGS["simulation_settings"]["dt0"])
        + "max_time_iter: {}\n".format(
            LINEAR_ODE_SYSTEM_SETTINGS["simulation_settings"]["max_time_iter"]
        )
        + "simulation data dimensions: {}\n".format(
            (LINEAR_ODE_SYSTEM_SETTINGS["simulation_settings"]["max_time_iter"], 2)
        )
        + "\n"
        + "parameter values:\n"
    )
    for key, value in LINEAR_ODE_SYSTEM_SETTINGS["hidden_params"].items():
        if value:
            expected_output += "{}: {} -> boundary\n".format(
                key[len("sample_") :],
                LINEAR_ODE_SYSTEM_SETTINGS["sample_boundaries"][key[len("sample_") :]],
            )
        else:
            expected_output += "{}: {} -> constant\n".format(
                key[len("sample_") :],
                LINEAR_ODE_SYSTEM_SETTINGS["default_param_values"][
                    key[len("sample_") :]
                ],
            )
    assert test_object.hidden_params == LINEAR_ODE_SYSTEM_SETTINGS["hidden_params"]
    assert (
        test_object.simulation_settings
        == LINEAR_ODE_SYSTEM_SETTINGS["simulation_settings"]
    )
    assert (
        test_object.sample_boundaries == LINEAR_ODE_SYSTEM_SETTINGS["sample_boundaries"]
    )
    assert (
        test_object.default_param_values
        == LINEAR_ODE_SYSTEM_SETTINGS["default_param_values"]
    )
    assert isinstance(test_object.dt0, float)
    assert test_object.dt0 == LINEAR_ODE_SYSTEM_SETTINGS["simulation_settings"]["dt0"]
    assert isinstance(test_object.max_time_iter, int)
    assert (
        test_object.max_time_iter
        == LINEAR_ODE_SYSTEM_SETTINGS["simulation_settings"]["max_time_iter"]
    )
    assert isinstance(test_object.is_pde, bool)
    assert not test_object.is_pde
    assert isinstance(test_object.t, np.ndarray)
    assert np.array_equal(test_object.t, test_object.get_time_points())
    assert isinstance(test_object.hidden_param_names, list)
    assert test_object.hidden_param_names == test_object.get_param_names()
    assert isinstance(test_object.default_param_kwargs, dict)
    assert test_object.default_param_kwargs == test_object.get_default_param_kwargs()
    assert test_object.num_features == 2
    assert out == expected_output
    assert err == ""


# -------------------------- Test inherited methods -------------------------- #


def test_linear_ode_system_get_time_points_method():
    test_object = LinearODEsystem(**LINEAR_ODE_SYSTEM_SETTINGS)
    time_points = test_object.get_time_points()
    assert isinstance(time_points, np.ndarray)
    assert len(time_points.shape) == 1
    assert (
        time_points.shape[0]
        == LINEAR_ODE_SYSTEM_SETTINGS["simulation_settings"]["max_time_iter"]
    )
    assert np.all(
        np.isclose(
            np.diff(time_points),
            LINEAR_ODE_SYSTEM_SETTINGS["simulation_settings"]["dt0"],
        )
    )


def test_linear_ode_system_get_param_names_method():
    test_object = LinearODEsystem(**LINEAR_ODE_SYSTEM_SETTINGS)
    hidden_param_names = test_object.get_param_names()
    assert isinstance(hidden_param_names, list)
    assert len(hidden_param_names) == sum(
        LINEAR_ODE_SYSTEM_SETTINGS["hidden_params"].values()
    )
    for key, value in LINEAR_ODE_SYSTEM_SETTINGS["hidden_params"].items():
        if value:
            assert key[len("sample_") :] in hidden_param_names
        else:
            assert key[len("sample_") :] not in hidden_param_names


def test_linear_ode_system_get_default_param_kwargs_method():
    test_object = LinearODEsystem(**LINEAR_ODE_SYSTEM_SETTINGS)
    default_param_kwargs = test_object.get_default_param_kwargs()
    assert isinstance(default_param_kwargs, dict)
    assert len(default_param_kwargs) == (
        len(LINEAR_ODE_SYSTEM_SETTINGS["hidden_params"])
        - sum(LINEAR_ODE_SYSTEM_SETTINGS["hidden_params"].values())
    )
    for key, value in LINEAR_ODE_SYSTEM_SETTINGS["hidden_params"].items():
        if not value:
            assert key[len("sample_") :] in default_param_kwargs
            assert (
                default_param_kwargs[key[len("sample_") :]]
                == LINEAR_ODE_SYSTEM_SETTINGS["default_param_values"][
                    key[len("sample_") :]
                ]
            )
        else:
            assert key[len("sample_") :] not in default_param_kwargs


def test_linear_ode_system_print_internal_settings_method(capsys):
    test_object = LinearODEsystem(**LINEAR_ODE_SYSTEM_SETTINGS)
    _, _ = capsys.readouterr()

    expected_output = (
        80 * "#"
        + "\n"
        + "\n"
        + "Initialize simulation model: LinearODEsystem\n"
        + 80 * "-"
        + "\n"
        + "hidden parameters: {}\n".format(test_object.hidden_param_names)
        + "dt0: {}\n".format(LINEAR_ODE_SYSTEM_SETTINGS["simulation_settings"]["dt0"])
        + "max_time_iter: {}\n".format(
            LINEAR_ODE_SYSTEM_SETTINGS["simulation_settings"]["max_time_iter"]
        )
        + "simulation data dimensions: {}\n".format(
            (LINEAR_ODE_SYSTEM_SETTINGS["simulation_settings"]["max_time_iter"], 2)
        )
        + "\n"
        + "parameter values:\n"
    )
    for key, value in LINEAR_ODE_SYSTEM_SETTINGS["hidden_params"].items():
        if value:
            expected_output += "{}: {} -> boundary\n".format(
                key[len("sample_") :],
                LINEAR_ODE_SYSTEM_SETTINGS["sample_boundaries"][key[len("sample_") :]],
            )
        else:
            expected_output += "{}: {} -> constant\n".format(
                key[len("sample_") :],
                LINEAR_ODE_SYSTEM_SETTINGS["default_param_values"][
                    key[len("sample_") :]
                ],
            )

    test_object.print_internal_settings()
    out, err = capsys.readouterr()
    assert out == expected_output
    assert err == ""


def test_linear_ode_system_sample_to_kwargs_method():
    dummy_sample = np.array(
        [
            random.uniform(-1000, 1000)
            for x in range(sum(LINEAR_ODE_SYSTEM_SETTINGS["hidden_params"].values()))
        ]
    ).astype(np.float32)
    test_object = LinearODEsystem(**LINEAR_ODE_SYSTEM_SETTINGS)
    param_kwargs = test_object.sample_to_kwargs(dummy_sample)
    assert isinstance(param_kwargs, dict)
    assert len(param_kwargs) == len(LINEAR_ODE_SYSTEM_SETTINGS["hidden_params"])
    assert len(param_kwargs) == len(LINEAR_ODE_SYSTEM_SETTINGS["sample_boundaries"])
    assert len(param_kwargs) == len(LINEAR_ODE_SYSTEM_SETTINGS["default_param_values"])
    counter = 0
    for key, value in param_kwargs.items():
        if LINEAR_ODE_SYSTEM_SETTINGS["hidden_params"]["sample_" + key]:
            assert value == dummy_sample[counter]
            counter += 1
        else:
            assert value == dummy_sample[key]


def test_linear_ode_system_uniform_prior_method():
    test_object = LinearODEsystem(**LINEAR_ODE_SYSTEM_SETTINGS)
    sample = test_object.uniform_prior(reject_sampling=False)
    # sample_reject = test_object.uniform_prior(reject_sampling=True)

    assert isinstance(sample, np.ndarray)
    assert sample.dtype == np.float32
    assert len(sample.shape) == 1
    assert sample.shape[0] == sum(LINEAR_ODE_SYSTEM_SETTINGS["hidden_params"].values())
    counter = 0
    for name, (lower_boundary, upper_boundary) in LINEAR_ODE_SYSTEM_SETTINGS[
        "sample_boundaries"
    ].items():
        if LINEAR_ODE_SYSTEM_SETTINGS["hidden_params"]["sample_" + name]:
            assert sample[counter] >= lower_boundary
            assert sample[counter] <= upper_boundary
            counter += 1

    # TODO: unit tests for reject_sampling=True


# --------------------- Test implemented abstract methods -------------------- #


def test_linear_ode_system_get_sim_data_sim_method():
    test_object = LinearODEsystem(**LINEAR_ODE_SYSTEM_SETTINGS)
    sim_data_dim = test_object.get_sim_data_dim()
    assert isinstance(sim_data_dim, tuple)
    assert len(sim_data_dim) == 2
    assert (
        sim_data_dim[0]
        == LINEAR_ODE_SYSTEM_SETTINGS["simulation_settings"]["max_time_iter"]
    )
    assert sim_data_dim[1] == 2


@pytest.mark.parametrize("dummy_matrix", dummy_matrices)
def test_linear_ode_system_reject_sampler_method(dummy_matrix):
    test_object = LinearODEsystem(**LINEAR_ODE_SYSTEM_SETTINGS)
    sample = np.concatenate(
        (
            dummy_matrix[0],
            np.array([random.uniform(-1000, 1000), random.uniform(-1000, 1000)]),
        )
    )
    reject = test_object.reject_sampler(sample)
    assert reject == dummy_matrix[1]
