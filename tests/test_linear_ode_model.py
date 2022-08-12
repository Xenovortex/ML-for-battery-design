import os
import random

import numpy as np
import pytest
from bayesflow.forward_inference import GenerativeModel, Prior, Simulator
from matplotlib.axes import Axes
from matplotlib.figure import Figure

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
            (LINEAR_ODE_SYSTEM_SETTINGS["simulation_settings"]["max_time_iter"], 4)
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
    batch_size = random.randint(1, 8)
    data_dict = test_object.generative_model(batch_size=batch_size)
    prior_samples = data_dict["prior_draws"]
    sim_data = data_dict["sim_data"]
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
    assert test_object.num_features == 4
    assert test_object.plot_settings == LINEAR_ODE_SYSTEM_SETTINGS["plot_settings"]
    assert isinstance(test_object.prior, Prior)
    assert isinstance(test_object.simulator, Simulator)
    assert isinstance(test_object.generative_model, GenerativeModel)
    assert len(prior_samples.shape) == 2
    assert prior_samples.shape[0] == batch_size
    assert prior_samples.shape[1] == sum(
        LINEAR_ODE_SYSTEM_SETTINGS["hidden_params"].values()
    )
    assert len(sim_data.shape) == 3
    assert sim_data.shape[0] == batch_size
    assert (
        sim_data.shape[1]
        == LINEAR_ODE_SYSTEM_SETTINGS["simulation_settings"]["max_time_iter"]
    )
    assert isinstance(test_object.prior_means, np.ndarray)
    assert isinstance(test_object.prior_stds, np.ndarray)
    assert len(test_object.prior_means.shape) == 2
    assert test_object.prior_means.shape[0] == 1
    assert test_object.prior_means.shape[1] == sum(
        LINEAR_ODE_SYSTEM_SETTINGS["hidden_params"].values()
    )
    assert len(test_object.prior_stds.shape) == 2
    assert test_object.prior_stds.shape[0] == 1
    assert test_object.prior_stds.shape[1] == sum(
        LINEAR_ODE_SYSTEM_SETTINGS["hidden_params"].values()
    )
    assert sim_data.shape[2] == test_object.num_features
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
            (LINEAR_ODE_SYSTEM_SETTINGS["simulation_settings"]["max_time_iter"], 4)
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
    dummy_sample = np.random.uniform(
        -1000, 1000, size=sum(LINEAR_ODE_SYSTEM_SETTINGS["hidden_params"].values())
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
    sample_reject = test_object.uniform_prior(reject_sampling=True)

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

    assert isinstance(sample_reject, np.ndarray)
    assert sample_reject.dtype == np.float32
    assert len(sample_reject.shape) == 1
    assert sample_reject.shape[0] == sum(
        LINEAR_ODE_SYSTEM_SETTINGS["hidden_params"].values()
    )
    counter = 0
    for name, (lower_boundary, upper_boundary) in LINEAR_ODE_SYSTEM_SETTINGS[
        "sample_boundaries"
    ].items():
        if LINEAR_ODE_SYSTEM_SETTINGS["hidden_params"]["sample_" + name]:
            assert sample_reject[counter] >= lower_boundary
            assert sample_reject[counter] <= upper_boundary
            counter += 1


def test_linear_ode_system_get_bayesflow_amortizer():
    test_object = LinearODEsystem(**LINEAR_ODE_SYSTEM_SETTINGS)
    prior, simulator, generative_model = test_object.get_bayesflow_amortizer()
    batch_size = random.randint(1, 8)
    data_dict = generative_model(batch_size=batch_size)
    prior_samples = data_dict["prior_draws"]
    sim_data = data_dict["sim_data"]
    assert isinstance(prior, Prior)
    assert isinstance(simulator, Simulator)
    assert isinstance(generative_model, GenerativeModel)
    assert len(prior_samples.shape) == 2
    assert prior_samples.shape[0] == batch_size
    assert prior_samples.shape[1] == sum(
        LINEAR_ODE_SYSTEM_SETTINGS["hidden_params"].values()
    )
    assert len(sim_data.shape) == 3
    assert sim_data.shape[0] == batch_size
    assert (
        sim_data.shape[1]
        == LINEAR_ODE_SYSTEM_SETTINGS["simulation_settings"]["max_time_iter"]
    )
    assert sim_data.shape[2] == 4


def test_linear_ode_system_get_prior_means_stds():
    test_object = LinearODEsystem(**LINEAR_ODE_SYSTEM_SETTINGS)
    prior_means, prior_stds = test_object.get_prior_means_stds()

    assert isinstance(prior_means, np.ndarray)
    assert isinstance(prior_stds, np.ndarray)
    assert len(prior_means.shape) == 2
    assert prior_means.shape[0] == 1
    assert prior_means.shape[1] == sum(
        LINEAR_ODE_SYSTEM_SETTINGS["hidden_params"].values()
    )
    assert len(prior_stds.shape) == 2
    assert prior_stds.shape[0] == 1
    assert prior_stds.shape[1] == sum(
        LINEAR_ODE_SYSTEM_SETTINGS["hidden_params"].values()
    )


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
    assert sim_data_dim[1] == 4


@pytest.mark.parametrize("dummy_matrix", dummy_matrices)
def test_linear_ode_system_reject_sampler_method(dummy_matrix):
    test_object = LinearODEsystem(**LINEAR_ODE_SYSTEM_SETTINGS)
    sample = np.concatenate((dummy_matrix[0], np.random.uniform(-1000, 1000, size=2)))
    reject = test_object.reject_sampler(sample)
    assert reject == dummy_matrix[1]


def test_linear_ode_system_solver_method():
    test_object = LinearODEsystem(**LINEAR_ODE_SYSTEM_SETTINGS)
    params = test_object.uniform_prior()
    solution = test_object.solver(params)
    assert isinstance(solution, np.ndarray)
    assert solution.dtype == np.float32
    assert solution.shape == test_object.get_sim_data_dim()


def test_linear_ode_system_plot_sim_data_one_plot():
    init_data = LINEAR_ODE_SYSTEM_SETTINGS
    init_data["plot_settings"]["num_plots"] = 1
    init_data["plot_settings"]["show_title"] = True
    init_data["plot_settings"]["show_plot"] = True
    init_data["plot_settings"]["show_params"] = True
    init_data["plot_settings"]["show_eigen"] = True
    test_object = LinearODEsystem(**init_data)
    fig, ax, params, sim_data = test_object.plot_sim_data(filename="pytest")
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert isinstance(params, np.ndarray)
    assert isinstance(sim_data, np.ndarray)
    assert len(params.shape) == 2
    assert params.shape[0] == init_data["plot_settings"]["num_plots"]
    assert params.shape[1] == sum(init_data["hidden_params"].values())
    assert len(sim_data.shape) == 3
    assert sim_data.shape[0] == init_data["plot_settings"]["num_plots"]
    assert sim_data.shape[1] == init_data["simulation_settings"]["max_time_iter"]
    assert sim_data.shape[2] == test_object.num_features
    for i in range(test_object.num_features):
        x_plot, y_plot = ax.lines[i].get_xydata().T
        assert np.array_equal(x_plot, test_object.t)
        assert np.array_equal(y_plot, sim_data[0, :, i])
    assert os.path.exists("results/pytest/plots/pytest-sim_data.png")
    if os.path.exists("results/pytest/plots/pytest-sim_data.png"):
        os.remove("results/pytest/plots/pytest-sim_data.png")
    if os.path.exists("results/pytest/plots"):
        os.rmdir("results/pytest/plots")
    if os.path.exists("results/pytest"):
        os.rmdir("results/pytest")


def test_linear_ode_system_plot_sim_data_multiple_row_plots():
    init_data = LINEAR_ODE_SYSTEM_SETTINGS
    init_data["plot_settings"]["num_plots"] = 8
    init_data["plot_settings"]["show_title"] = True
    init_data["plot_settings"]["show_plot"] = True
    init_data["plot_settings"]["show_params"] = True
    init_data["plot_settings"]["show_eigen"] = True
    test_object = LinearODEsystem(**init_data)
    fig, ax, params, sim_data = test_object.plot_sim_data(filename="pytest")
    assert isinstance(fig, Figure)
    assert isinstance(ax, np.flatiter)
    assert isinstance(params, np.ndarray)
    assert isinstance(sim_data, np.ndarray)
    assert len(params.shape) == 2
    assert params.shape[0] == init_data["plot_settings"]["num_plots"]
    assert params.shape[1] == sum(init_data["hidden_params"].values())
    assert len(sim_data.shape) == 3
    assert sim_data.shape[0] == init_data["plot_settings"]["num_plots"]
    assert sim_data.shape[1] == init_data["simulation_settings"]["max_time_iter"]
    assert sim_data.shape[2] == test_object.num_features
    for k in range(init_data["plot_settings"]["num_plots"]):
        for i in range(test_object.num_features):
            x_plot, y_plot = ax[k].lines[i].get_xydata().T
            assert np.array_equal(x_plot, test_object.t)
            assert np.array_equal(y_plot, sim_data[k, :, i])
    assert os.path.exists("results/pytest/plots/pytest-sim_data.png")
    if os.path.exists("results/pytest/plots/pytest-sim_data.png"):
        os.remove("results/pytest/plots/pytest-sim_data.png")
    if os.path.exists("results/pytest/plots"):
        os.rmdir("results/pytest/plots")
    if os.path.exists("results/pytest"):
        os.rmdir("results/pytest")
