import os
import random

import numpy as np
import pytest
from bayesflow.forward_inference import GenerativeModel, Prior, Simulator
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ML_for_Battery_Design.src.settings.SPM_battery_settings import (
    SPM_BATTERY_MODEL_SIMULATION_SETTINGS,
)
from ML_for_Battery_Design.src.simulation.SPM_battery import SPMBatteryModel
from tests.constants import AUTO_CLOSE_PLOTS

# ------------------- Test SPM Battery Class Initialization ------------------ #


def test_spm_battery_init(capsys):
    init_data = SPM_BATTERY_MODEL_SIMULATION_SETTINGS
    expected_num_features = 1
    test_object = SPMBatteryModel(**init_data)
    out, err = capsys.readouterr()
    expected_output = (
        80 * "#"
        + "\n"
        + "\n"
        + "Initialize simulation model: SPMBatteryModel\n"
        + 80 * "-"
        + "\n"
        + "hidden parameters: {}\n".format(test_object.hidden_param_names)
        + "dt0: {}\n".format(init_data["simulation_settings"]["dt0"])
        + "max_time_iter: {}\n".format(
            init_data["simulation_settings"]["max_time_iter"]
        )
        + "nr: {}\n".format(init_data["simulation_settings"]["nr"])
        + "simulation data dimensions: {}\n".format(
            (
                init_data["simulation_settings"]["max_time_iter"],
                init_data["simulation_settings"]["nr"],
                expected_num_features,
            )
        )
        + "\n"
        + "parameter values:\n"
    )
    for key, value in init_data["hidden_params"].items():
        if value:
            expected_output += "{}: {} -> boundary\n".format(
                key[len("sample_") :],
                init_data["sample_boundaries"][key[len("sample_") :]],
            )
        else:
            expected_output += "{}: {} -> constant\n".format(
                key[len("sample_") :],
                init_data["default_param_values"][key[len("sample_") :]],
            )
    batch_size = random.randint(1, 8)
    data_dict = test_object.generative_model(batch_size=batch_size)
    prior_samples = data_dict["prior_draws"]
    sim_data = data_dict["sim_data"]
    assert isinstance(test_object.hidden_params, dict)
    assert isinstance(test_object.simulation_settings, dict)
    assert isinstance(test_object.sample_boundaries, dict)
    assert isinstance(test_object.default_param_values, dict)
    assert isinstance(test_object.plot_settings, dict)
    assert test_object.hidden_params == init_data["hidden_params"]
    assert test_object.simulation_settings == init_data["simulation_settings"]
    assert test_object.sample_boundaries == init_data["sample_boundaries"]
    assert test_object.default_param_values == init_data["default_param_values"]
    assert test_object.plot_settings == init_data["plot_settings"]
    assert isinstance(test_object.dt0, float)
    assert test_object.dt0 == init_data["simulation_settings"]["dt0"]
    assert isinstance(test_object.max_time_iter, int)
    assert (
        test_object.max_time_iter == init_data["simulation_settings"]["max_time_iter"]
    )
    assert test_object.nr == init_data["simulation_settings"]["nr"]
    assert isinstance(test_object.reject_sampling, bool)
    assert (
        test_object.reject_sampling
        == init_data["simulation_settings"]["use_reject_sampling"]
    )
    assert isinstance(test_object.is_pde, bool)
    assert test_object.is_pde
    assert isinstance(test_object.t, np.ndarray)
    assert np.array_equal(test_object.t, test_object.get_time_points())
    assert isinstance(test_object.hidden_param_names, list)
    assert test_object.hidden_param_names == test_object.get_param_names()
    assert isinstance(test_object.num_hidden_params, int)
    assert test_object.num_hidden_params == sum(init_data["hidden_params"].values())
    assert isinstance(test_object.default_param_kwargs, dict)
    assert test_object.default_param_kwargs == test_object.get_default_param_kwargs()
    assert isinstance(test_object.prior, Prior)
    assert isinstance(test_object.simulator, Simulator)
    assert isinstance(test_object.generative_model, GenerativeModel)
    assert prior_samples.ndim == 2
    assert prior_samples.shape[0] == batch_size
    assert prior_samples.shape[1] == sum(init_data["hidden_params"].values())
    assert sim_data.ndim == 4
    assert sim_data.shape[0] == batch_size
    assert sim_data.shape[1] == init_data["simulation_settings"]["max_time_iter"]
    assert sim_data.shape[2] == init_data["simulation_settings"]["nr"]
    assert sim_data.shape[3] == test_object.num_features
    assert isinstance(test_object.prior_means, np.ndarray)
    assert isinstance(test_object.prior_stds, np.ndarray)
    assert test_object.prior_means.ndim == 2
    assert test_object.prior_means.shape[0] == 1
    assert test_object.prior_means.shape[1] == sum(init_data["hidden_params"].values())
    assert test_object.prior_stds.ndim == 2
    assert test_object.prior_stds.shape[0] == 1
    assert test_object.prior_stds.shape[1] == sum(init_data["hidden_params"].values())
    assert out == expected_output
    assert err == ""


# -------------------------- Test inherited methods -------------------------- #


def test_spm_battery_get_time_points_method():
    test_object = SPMBatteryModel(**SPM_BATTERY_MODEL_SIMULATION_SETTINGS)
    time_points = test_object.get_time_points()

    assert isinstance(time_points, np.ndarray)
    assert time_points.ndim == 1
    assert (
        time_points.shape[0]
        == SPM_BATTERY_MODEL_SIMULATION_SETTINGS["simulation_settings"]["max_time_iter"]
    )
    assert np.all(
        np.isclose(
            np.diff(time_points),
            SPM_BATTERY_MODEL_SIMULATION_SETTINGS["simulation_settings"]["dt0"],
        )
    )


def test_spm_battery_get_param_names_method():
    test_object = SPMBatteryModel(**SPM_BATTERY_MODEL_SIMULATION_SETTINGS)
    hidden_param_names = test_object.get_param_names()

    assert isinstance(hidden_param_names, list)
    assert len(hidden_param_names) == sum(
        SPM_BATTERY_MODEL_SIMULATION_SETTINGS["hidden_params"].values()
    )
    for key, value in SPM_BATTERY_MODEL_SIMULATION_SETTINGS["hidden_params"].items():
        if value:
            assert key[len("sample_") :] in hidden_param_names
        else:
            assert key[len("sample_") :] not in hidden_param_names


def test_spm_battery_get_default_param_kwargs_method():
    test_object = SPMBatteryModel(**SPM_BATTERY_MODEL_SIMULATION_SETTINGS)
    default_param_kwargs = test_object.get_default_param_kwargs()

    assert isinstance(default_param_kwargs, dict)
    assert len(default_param_kwargs) == (
        len(SPM_BATTERY_MODEL_SIMULATION_SETTINGS["hidden_params"])
        - sum(SPM_BATTERY_MODEL_SIMULATION_SETTINGS["hidden_params"].values())
    )
    for key, value in SPM_BATTERY_MODEL_SIMULATION_SETTINGS["hidden_params"].items():
        if not value:
            assert key[len("sample_") :] in default_param_kwargs
            assert (
                default_param_kwargs[key[len("sample_") :]]
                == SPM_BATTERY_MODEL_SIMULATION_SETTINGS["default_param_values"][
                    key[len("sample_") :]
                ]
            )
        else:
            assert key[len("sample_") :] not in default_param_kwargs


def test_spm_battery_print_internal_settings_method(capsys):
    init_data = SPM_BATTERY_MODEL_SIMULATION_SETTINGS
    expected_num_features = 1
    test_object = SPMBatteryModel(**init_data)
    _, _ = capsys.readouterr()

    expected_output = (
        80 * "#"
        + "\n"
        + "\n"
        + "Initialize simulation model: SPMBatteryModel\n"
        + 80 * "-"
        + "\n"
        + "hidden parameters: {}\n".format(test_object.hidden_param_names)
        + "dt0: {}\n".format(init_data["simulation_settings"]["dt0"])
        + "max_time_iter: {}\n".format(
            init_data["simulation_settings"]["max_time_iter"]
        )
        + "nr: {}\n".format(init_data["simulation_settings"]["nr"])
        + "simulation data dimensions: {}\n".format(
            (
                init_data["simulation_settings"]["max_time_iter"],
                init_data["simulation_settings"]["nr"],
                expected_num_features,
            )
        )
        + "\n"
        + "parameter values:\n"
    )
    for key, value in init_data["hidden_params"].items():
        if value:
            expected_output += "{}: {} -> boundary\n".format(
                key[len("sample_") :],
                init_data["sample_boundaries"][key[len("sample_") :]],
            )
        else:
            expected_output += "{}: {} -> constant\n".format(
                key[len("sample_") :],
                init_data["default_param_values"][key[len("sample_") :]],
            )

    test_object.print_internal_settings()
    out, err = capsys.readouterr()
    assert out == expected_output
    assert err == ""


def test_spm_battery_sample_to_kwargs_method():
    dummy_sample = np.random.uniform(
        -1000,
        1000,
        size=sum(SPM_BATTERY_MODEL_SIMULATION_SETTINGS["hidden_params"].values()),
    )
    test_object = SPMBatteryModel(**SPM_BATTERY_MODEL_SIMULATION_SETTINGS)
    param_kwargs = test_object.sample_to_kwargs(dummy_sample)

    assert isinstance(param_kwargs, dict)
    assert len(param_kwargs) == len(
        SPM_BATTERY_MODEL_SIMULATION_SETTINGS["hidden_params"]
    )
    assert len(param_kwargs) == len(
        SPM_BATTERY_MODEL_SIMULATION_SETTINGS["sample_boundaries"]
    )
    assert len(param_kwargs) == len(
        SPM_BATTERY_MODEL_SIMULATION_SETTINGS["default_param_values"]
    )
    counter = 0
    for key, value in param_kwargs.items():
        if SPM_BATTERY_MODEL_SIMULATION_SETTINGS["hidden_params"]["sample_" + key]:
            assert value == dummy_sample[counter]
            counter += 1
        else:
            assert (
                value
                == SPM_BATTERY_MODEL_SIMULATION_SETTINGS["default_param_values"][key]
            )


def test_spm_battery_uniform_prior_method():
    test_object = SPMBatteryModel(**SPM_BATTERY_MODEL_SIMULATION_SETTINGS)
    sample = test_object.uniform_prior()

    assert isinstance(sample, np.ndarray)
    assert sample.ndim == 1
    assert sample.shape[0] == sum(
        SPM_BATTERY_MODEL_SIMULATION_SETTINGS["hidden_params"].values()
    )
    counter = 0
    for name, (lower_boundary, upper_boundary) in SPM_BATTERY_MODEL_SIMULATION_SETTINGS[
        "sample_boundaries"
    ].items():
        if SPM_BATTERY_MODEL_SIMULATION_SETTINGS["hidden_params"]["sample_" + name]:
            assert sample[counter] >= lower_boundary
            assert sample[counter] <= upper_boundary
            counter += 1


def test_spm_battery_get_bayesflow_generator():
    test_object = SPMBatteryModel(**SPM_BATTERY_MODEL_SIMULATION_SETTINGS)
    prior, simulator, generative_model = test_object.get_bayesflow_generator()
    batch_size = random.randint(1, 8)
    data_dict = generative_model(batch_size=batch_size)
    prior_samples = data_dict["prior_draws"]
    sim_data = data_dict["sim_data"]

    assert isinstance(prior, Prior)
    assert isinstance(simulator, Simulator)
    assert isinstance(generative_model, GenerativeModel)
    assert prior_samples.ndim == 2
    assert prior_samples.shape[0] == batch_size
    assert prior_samples.shape[1] == sum(
        SPM_BATTERY_MODEL_SIMULATION_SETTINGS["hidden_params"].values()
    )
    assert sim_data.ndim == 4
    assert sim_data.shape[0] == batch_size
    assert (
        sim_data.shape[1]
        == SPM_BATTERY_MODEL_SIMULATION_SETTINGS["simulation_settings"]["max_time_iter"]
    )
    assert (
        sim_data.shape[2]
        == SPM_BATTERY_MODEL_SIMULATION_SETTINGS["simulation_settings"]["nr"]
    )
    assert sim_data.shape[3] == 1


def test_spm_battery_get_prior_means_stds():
    test_object = SPMBatteryModel(**SPM_BATTERY_MODEL_SIMULATION_SETTINGS)
    prior_means, prior_stds = test_object.get_prior_means_stds()

    assert isinstance(prior_means, np.ndarray)
    assert isinstance(prior_stds, np.ndarray)
    assert prior_means.ndim == 2
    assert prior_means.shape[0] == 1
    assert prior_means.shape[1] == sum(
        SPM_BATTERY_MODEL_SIMULATION_SETTINGS["hidden_params"].values()
    )
    assert prior_stds.ndim == 2
    assert prior_stds.shape[0] == 1
    assert prior_stds.shape[1] == sum(
        SPM_BATTERY_MODEL_SIMULATION_SETTINGS["hidden_params"].values()
    )


# --------------------- Test implemented abstract methods -------------------- #


def test_spm_battery_get_sim_data_shape_method():
    init_data = SPM_BATTERY_MODEL_SIMULATION_SETTINGS
    test_object = SPMBatteryModel(**init_data)
    sim_data_dim = test_object.get_sim_data_shape()

    assert isinstance(sim_data_dim, tuple)
    assert len(sim_data_dim) == 3
    assert (
        sim_data_dim[0]
        == SPM_BATTERY_MODEL_SIMULATION_SETTINGS["simulation_settings"]["max_time_iter"]
    )
    assert (
        sim_data_dim[1]
        == SPM_BATTERY_MODEL_SIMULATION_SETTINGS["simulation_settings"]["nr"]
    )
    assert sim_data_dim[2] == 1


def test_spm_battery_solver_method():
    init_data = SPM_BATTERY_MODEL_SIMULATION_SETTINGS
    test_object = SPMBatteryModel(**init_data)
    params = test_object.uniform_prior()
    solution = test_object.solver(params)

    assert isinstance(solution, np.ndarray)
    assert solution.shape == test_object.get_sim_data_shape()
    assert np.all(np.isfinite(solution))


def test_spm_battery_plot_sim_data_one_plot():
    init_data = SPM_BATTERY_MODEL_SIMULATION_SETTINGS
    init_data["plot_settings"]["num_plots"] = 1
    init_data["plot_settings"]["show_title"] = True
    init_data["plot_settings"]["show_plot"] = True
    init_data["plot_settings"]["show_time"] = 0 if AUTO_CLOSE_PLOTS else None

    test_object = SPMBatteryModel(**init_data)
    fig, ax, params, sim_data = test_object.plot_sim_data(parent_folder="pytest")

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert isinstance(params, np.ndarray)
    assert isinstance(sim_data, np.ndarray)
    assert params.ndim == 2
    assert params.shape[0] == init_data["plot_settings"]["num_plots"]
    assert params.shape[1] == sum(init_data["hidden_params"].values())
    assert sim_data.ndim == 4
    assert sim_data.shape[0] == init_data["plot_settings"]["num_plots"]
    assert sim_data.shape[1] == init_data["simulation_settings"]["max_time_iter"]
    assert sim_data.shape[2] == init_data["simulation_settings"]["nr"]
    assert sim_data.shape[3] == test_object.num_features
    assert os.path.exists(os.path.join("pytest", "sim_data.png"))
    if os.path.exists(os.path.join("pytest", "sim_data.png")):
        os.remove(os.path.join("pytest", "sim_data.png"))
    if os.path.exists("pytest"):
        os.rmdir("pytest")


def test_spm_battery_plot_sim_data_multiple_row_plots():
    init_data = SPM_BATTERY_MODEL_SIMULATION_SETTINGS
    init_data["plot_settings"]["num_plots"] = 8
    init_data["plot_settings"]["show_title"] = True
    init_data["plot_settings"]["show_plot"] = True
    init_data["plot_settings"]["show_time"] = 0 if AUTO_CLOSE_PLOTS else None
    test_object = SPMBatteryModel(**init_data)
    fig, ax, params, sim_data = test_object.plot_sim_data(parent_folder="pytest")

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert isinstance(params, np.ndarray)
    assert isinstance(sim_data, np.ndarray)
    assert params.ndim == 2
    assert params.shape[0] == init_data["plot_settings"]["num_plots"]
    assert params.shape[1] == sum(init_data["hidden_params"].values())
    assert sim_data.ndim == 4
    assert sim_data.shape[0] == init_data["plot_settings"]["num_plots"]
    assert sim_data.shape[1] == init_data["simulation_settings"]["max_time_iter"]
    assert sim_data.shape[2] == init_data["simulation_settings"]["nr"]
    assert sim_data.shape[3] == test_object.num_features
    assert os.path.exists(os.path.join("pytest", "sim_data.png"))
    if os.path.exists(os.path.join("pytest", "sim_data.png")):
        os.remove(os.path.join("pytest", "sim_data.png"))
    if os.path.exists("pytest"):
        os.rmdir("pytest")


def test_spm_battery_plot_sim_data_negative_num_plots(capsys):
    init_data = SPM_BATTERY_MODEL_SIMULATION_SETTINGS
    init_data["plot_settings"]["num_plots"] = random.randint(-10, 0)
    test_object = SPMBatteryModel(**init_data)
    with pytest.raises(ValueError):
        test_object.plot_sim_data()
        out, err = capsys.readouterr()
        assert out == ""
        assert (
            err
            == "{} - plot_sim_data: num_plots is {}, but can not be negative or zero".format(
                test_object.__class__.__name__, init_data["plot_settings"]["num_plots"]
            )
        )


def test_spm_battery_plot_resimulation_one_plot():
    init_data = SPM_BATTERY_MODEL_SIMULATION_SETTINGS
    init_data["plot_settings"]["num_plots"] = 1
    init_data["plot_settings"]["show_title"] = True
    init_data["plot_settings"]["show_plot"] = True
    init_data["plot_settings"]["show_time"] = 0 if AUTO_CLOSE_PLOTS else None

    test_object = SPMBatteryModel(**init_data)
    n_samples = random.randint(1, 100)
    dummy_post_samples = np.empty(
        (
            init_data["plot_settings"]["num_plots"],
            n_samples,
            test_object.num_hidden_params,
        )
    )
    for j in range(n_samples):
        dummy_post_samples[0, j, :] = test_object.uniform_prior()
    dummy_resimulation = np.empty(
        tuple(
            [init_data["plot_settings"]["num_plots"], n_samples]
            + list(test_object.get_sim_data_shape())
        )
    )
    for j in range(n_samples):
        dummy_resimulation[0, j] = test_object.solver(dummy_post_samples[0, j, :])
    dummy_ground_truth = np.median(dummy_resimulation, axis=1)
    fig, ax, resim_data = test_object.plot_resimulation(
        dummy_post_samples, dummy_ground_truth, parent_folder="pytest"
    )
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert isinstance(resim_data, np.ndarray)
    assert resim_data.ndim == 5
    assert resim_data.shape[0] == init_data["plot_settings"]["num_plots"]
    assert resim_data.shape[1] == n_samples
    assert resim_data.shape[2] == test_object.max_time_iter
    assert resim_data.shape[3] == test_object.nr
    assert resim_data.shape[4] == test_object.num_features
    for i in range(init_data["simulation_settings"]["nr"]):
        assert os.path.exists(os.path.join("pytest", "resimulation_nr{}.png".format(i)))
        if os.path.exists(os.path.join("pytest", "resimulation_nr{}.png".format(i))):
            os.remove(os.path.join("pytest", "resimulation_nr{}.png".format(i)))
    if os.path.exists("pytest"):
        os.rmdir("pytest")


def test_spm_battery_plot_resimulation_multiple_plots():
    init_data = SPM_BATTERY_MODEL_SIMULATION_SETTINGS
    init_data["plot_settings"]["num_plots"] = 8
    init_data["plot_settings"]["show_title"] = True
    init_data["plot_settings"]["show_plot"] = True
    init_data["plot_settings"]["show_time"] = 0 if AUTO_CLOSE_PLOTS else None

    test_object = SPMBatteryModel(**init_data)
    n_samples = random.randint(1, 100)
    dummy_post_samples = np.empty(
        (
            init_data["plot_settings"]["num_plots"],
            n_samples,
            test_object.num_hidden_params,
        )
    )
    for i in range(init_data["plot_settings"]["num_plots"]):
        for j in range(n_samples):
            dummy_post_samples[i, j, :] = test_object.uniform_prior()
    dummy_resimulation = np.empty(
        tuple(
            [init_data["plot_settings"]["num_plots"], n_samples]
            + list(test_object.get_sim_data_shape())
        )
    )
    for i in range(init_data["plot_settings"]["num_plots"]):
        for j in range(n_samples):
            dummy_resimulation[i, j] = test_object.solver(dummy_post_samples[i, j, :])
    dummy_ground_truth = np.median(dummy_resimulation, axis=1)

    fig, ax, resim_data = test_object.plot_resimulation(
        dummy_post_samples, dummy_ground_truth, parent_folder="pytest"
    )

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert isinstance(resim_data, np.ndarray)
    assert resim_data.ndim == 5
    assert resim_data.shape[0] == init_data["plot_settings"]["num_plots"]
    assert resim_data.shape[1] == n_samples
    assert resim_data.shape[2] == test_object.max_time_iter
    assert resim_data.shape[3] == test_object.nr
    assert resim_data.shape[4] == test_object.num_features
    for i in range(init_data["simulation_settings"]["nr"]):
        assert os.path.exists(os.path.join("pytest", "resimulation_nr{}.png".format(i)))
        if os.path.exists(os.path.join("pytest", "resimulation_nr{}.png".format(i))):
            os.remove(os.path.join("pytest", "resimulation_nr{}.png".format(i)))
    if os.path.exists("pytest"):
        os.rmdir("pytest")


def test_spm_battery_plot_resimulation_negative_num_plots(capsys):
    init_data = SPM_BATTERY_MODEL_SIMULATION_SETTINGS
    init_data["plot_settings"]["num_plots"] = random.randint(-10, 0)
    test_object = SPMBatteryModel(**init_data)
    n_samples = random.randint(1, 100)
    num_post_samples = random.randint(1, 4)
    dummy_post_samples = np.empty(
        (
            num_post_samples,
            n_samples,
            test_object.num_hidden_params,
        )
    )
    for i in range(init_data["plot_settings"]["num_plots"]):
        for j in range(n_samples):
            dummy_post_samples[i, j, :] = test_object.uniform_prior()
    dummy_ground_truth = np.random.uniform(
        -1000,
        1000,
        size=tuple([num_post_samples] + list(test_object.get_sim_data_shape())),
    )
    with pytest.raises(ValueError):
        test_object.plot_resimulation(dummy_post_samples, dummy_ground_truth)
        out, err = capsys.readouterr()
        assert out == ""
        assert (
            err
            == "{} - plot_resimulation: num_plots is {}, but can not be negative or zero".format(
                test_object.__class__.__name__, init_data["plot_settings"]["num_plots"]
            )
        )


def test_spm_battery_plot_resimulation_num_plots_larger_post_samples(capsys):
    init_data = SPM_BATTERY_MODEL_SIMULATION_SETTINGS
    init_data["plot_settings"]["num_plots"] = random.randint(5, 8)
    test_object = SPMBatteryModel(**init_data)
    n_samples = random.randint(1, 100)
    num_post_samples = random.randint(1, 4)
    dummy_post_samples = np.empty(
        (
            num_post_samples,
            n_samples,
            test_object.num_hidden_params,
        )
    )
    for i in range(num_post_samples):
        for j in range(n_samples):
            dummy_post_samples[i, j, :] = test_object.uniform_prior()
    dummy_ground_truth = np.random.uniform(
        -1000,
        1000,
        size=tuple([num_post_samples] + list(test_object.get_sim_data_shape())),
    )
    with pytest.raises(ValueError):
        test_object.plot_resimulation(dummy_post_samples, dummy_ground_truth)
        out, err = capsys.readouterr()
        assert out == ""
        assert (
            err
            == "{} - plot_resimulation: num_plots is {}, but only {} post_samples given".format(
                test_object.__class__.__name__,
                init_data["plot_settings"]["num_plots"],
                dummy_post_samples.shape[0],
            )
        )
