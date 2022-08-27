import os
import random

import numpy as np
import pandas as pd
import pytest
from bayesflow.amortized_inference import AmortizedPosterior
from bayesflow.helper_functions import build_meta_dict
from bayesflow.networks import InvertibleNetwork
from bayesflow.trainers import Trainer

from ML_for_Battery_Design.src.helpers.constants import (
    architecture_settings,
    inference_settings,
    sim_model_collection,
    simulation_settings,
    summary_collection,
)
from ML_for_Battery_Design.src.helpers.evaluater import Evaluater
from ML_for_Battery_Design.src.helpers.processing import Processing
from ML_for_Battery_Design.src.simulation.simulation_model import SimulationModel
from tests.constants import AUTO_CLOSE_PLOTS

models = ["linear_ode_system"]

non_dict_str_input = [
    random.randint(-1000, 1000),  # int
    random.uniform(-1000, 1000),  # float
    random.uniform(-1000, 1000) + random.uniform(-1000, 1000) * 1j,  # complex
    random.choice([True, False]),  # bool
    tuple(random.random() for x in range(random.randrange(10))),  # tuple
    list(random.random() for x in range(random.randrange(10))),  # list
    set(random.random() for x in range(random.randrange(10))),  # set
    frozenset(random.random() for x in range(random.randrange(10))),  # frozenset
    None,  # NoneType
]


def setup_dummy_objects(model_name, remove_nan=True):
    sim_model = sim_model_collection[model_name](**simulation_settings[model_name])
    summary_architecture = architecture_settings[model_name]["FC"]
    summary_net = summary_collection["FC"](build_meta_dict({}, summary_architecture))
    inn_architecture = architecture_settings[model_name]["INN"]
    inference_net = InvertibleNetwork(
        {**{"n_params": sim_model.num_hidden_params}, **inn_architecture}
    )
    dummy_amortizer = AmortizedPosterior(
        inference_net, summary_net, name="dummy_amortizer"
    )
    process_settings = inference_settings[model_name]["processing"]
    process_settings["remove_nan"] = remove_nan
    configurator = Processing(
        process_settings,
        sim_model.prior_means,
        sim_model.prior_stds,
    )
    dummy_trainer = Trainer(
        amortizer=dummy_amortizer,
        generative_model=sim_model.generative_model,
        configurator=configurator,
        learning_rate=1e-3,
    )
    return sim_model, dummy_amortizer, dummy_trainer


@pytest.mark.parametrize("model_name", models)
def test_evaluater_init(model_name):
    sim_model, dummy_amortizer, dummy_trainer = setup_dummy_objects(model_name)

    evaluater = Evaluater(
        sim_model,
        dummy_amortizer,
        dummy_trainer,
        simulation_settings[model_name]["plot_settings"],
        inference_settings[model_name]["evaluation"],
    )

    assert isinstance(evaluater.sim_model, SimulationModel)
    assert isinstance(evaluater.amortizer, AmortizedPosterior)
    assert isinstance(evaluater.trainer, Trainer)
    assert isinstance(evaluater.plot_settings, dict)
    assert isinstance(evaluater.eval_settings, dict)
    assert isinstance(evaluater.test_dict, dict)
    assert evaluater.plot_settings == simulation_settings[model_name]["plot_settings"]
    assert evaluater.eval_settings == inference_settings[model_name]["evaluation"]


@pytest.mark.parametrize("model_name", models)
def test_evaluater_evaluate_sim_model_plot_prior(model_name):
    sim_model, dummy_amortizer, dummy_trainer = setup_dummy_objects(model_name)

    dummy_plot_settings = simulation_settings[model_name]["plot_settings"]
    dummy_plot_settings["show_plot"] = True
    dummy_plot_settings["show_time"] = 0 if AUTO_CLOSE_PLOTS else None

    dummy_eval_settings = {
        "batch_size": random.randint(1, 8),
        "n_samples": random.randint(1, 8),
        "plot_prior": True,
        "plot_sim_data": False,
        "plot_loss": False,
        "plot_latent": False,
        "plot_sbc_histogram": False,
        "plot_sbc_ecdf": False,
        "plot_true_vs_estimated": False,
        "plot_posterior": False,
        "plot_post_with_prior": False,
        "plot_resimulation": False,
    }

    evaluater = Evaluater(
        sim_model,
        dummy_amortizer,
        dummy_trainer,
        dummy_plot_settings,
        dummy_eval_settings,
    )
    evaluater.evaluate_sim_model("pytest")

    assert os.path.exists(os.path.join("pytest", "prior_2d.png"))
    if os.path.exists(os.path.join("pytest", "prior_2d.png")):
        os.remove(os.path.join("pytest", "prior_2d.png"))
    if os.path.exists("pytest"):
        os.rmdir("pytest")


@pytest.mark.parametrize("model_name", models)
def test_evaluater_evaluate_sim_model_plot_sim_data(model_name):
    sim_model, dummy_amortizer, dummy_trainer = setup_dummy_objects(model_name)

    dummy_plot_settings = simulation_settings[model_name]["plot_settings"]
    dummy_plot_settings["show_plot"] = True
    dummy_plot_settings["show_time"] = 0 if AUTO_CLOSE_PLOTS else None

    dummy_eval_settings = {
        "batch_size": random.randint(1, 8),
        "n_samples": random.randint(1, 8),
        "plot_prior": False,
        "plot_sim_data": True,
        "plot_loss": False,
        "plot_latent": False,
        "plot_sbc_histogram": False,
        "plot_sbc_ecdf": False,
        "plot_true_vs_estimated": False,
        "plot_posterior": False,
        "plot_post_with_prior": False,
        "plot_resimulation": False,
    }

    evaluater = Evaluater(
        sim_model,
        dummy_amortizer,
        dummy_trainer,
        dummy_plot_settings,
        dummy_eval_settings,
    )
    evaluater.evaluate_sim_model("pytest")

    assert os.path.exists(os.path.join("pytest", "sim_data.png"))
    if os.path.exists(os.path.join("pytest", "sim_data.png")):
        os.remove(os.path.join("pytest", "sim_data.png"))
    if os.path.exists("pytest"):
        os.rmdir("pytest")


@pytest.mark.parametrize("model_name", models)
def test_evaluater_load_losses_dataframe(model_name):
    sim_model, dummy_amortizer, dummy_trainer = setup_dummy_objects(model_name)
    num_iterations = random.randint(1, 10)
    dummy_losses = pd.DataFrame(
        np.random.uniform(-100, 100, size=(num_iterations, 1)), columns=["Default.Loss"]
    )

    evaluater = Evaluater(
        sim_model,
        dummy_amortizer,
        dummy_trainer,
        simulation_settings[model_name]["plot_settings"],
        inference_settings[model_name]["evaluation"],
    )

    evaluater.load_losses(dummy_losses)

    assert isinstance(evaluater.losses, pd.DataFrame)
    assert evaluater.losses.equals(dummy_losses)


@pytest.mark.parametrize("model_name", models)
def test_evaluater_load_losses_path(model_name):
    sim_model, dummy_amortizer, dummy_trainer = setup_dummy_objects(model_name)
    num_iterations = random.randint(1, 10)
    dummy_losses = pd.DataFrame(
        np.random.uniform(-100, 100, size=(num_iterations, 1)), columns=["Default.Loss"]
    )

    if not os.path.exists("pytest"):
        os.mkdir("pytest")
    dummy_losses.to_pickle(os.path.join("pytest", "losses.pickle"))

    evaluater = Evaluater(
        sim_model,
        dummy_amortizer,
        dummy_trainer,
        simulation_settings[model_name]["plot_settings"],
        inference_settings[model_name]["evaluation"],
    )

    evaluater.load_losses(os.path.join("pytest", "losses.pickle"))

    assert isinstance(evaluater.losses, pd.DataFrame)
    assert evaluater.losses.equals(dummy_losses)

    if os.path.exists(os.path.join("pytest", "losses.pickle")):
        os.remove(os.path.join("pytest", "losses.pickle"))
    if os.path.exists("pytest"):
        os.rmdir("pytest")


@pytest.mark.parametrize("model_name", models)
@pytest.mark.parametrize("dummy_loss", non_dict_str_input)
def test_evaluater_load_losses_type_error(model_name, dummy_loss, capsys):
    sim_model, dummy_amortizer, dummy_trainer = setup_dummy_objects(model_name)

    evaluater = Evaluater(
        sim_model,
        dummy_amortizer,
        dummy_trainer,
        simulation_settings[model_name]["plot_settings"],
        inference_settings[model_name]["evaluation"],
    )

    with pytest.raises(TypeError):
        evaluater.load_losses(dummy_loss)
        out, err = capsys.readouterr()
        assert out == ""
        assert (
            err
            == "{} - load_losses: argument losses is {}, but has to be string or dict".format(
                evaluater.__class__.__name__, type(dummy_loss)
            )
        )


@pytest.mark.parametrize("model_name", models)
def test_evaluater_generate_test_data(model_name):
    sim_model, dummy_amortizer, dummy_trainer = setup_dummy_objects(
        model_name, remove_nan=False
    )
    batch_size = random.randint(2, 8)
    n_samples = random.randint(2, 8)

    evaluater = Evaluater(
        sim_model,
        dummy_amortizer,
        dummy_trainer,
        simulation_settings[model_name]["plot_settings"],
        inference_settings[model_name]["evaluation"],
    )

    test_dict = evaluater.generate_test_data(batch_size, n_samples)

    assert isinstance(test_dict, dict)
    assert isinstance(test_dict["test_data_raw"], dict)
    assert isinstance(test_dict["test_data_process"], dict)
    assert isinstance(test_dict["posterior_samples"], np.ndarray)
    assert isinstance(test_dict["posterior_samples_unnorm"], np.ndarray)
    assert test_dict["test_data_raw"]["prior_draws"].ndim == 2
    assert test_dict["test_data_raw"]["prior_draws"].shape[0] == batch_size
    assert (
        test_dict["test_data_raw"]["prior_draws"].shape[1]
        == sim_model.num_hidden_params
    )
    assert test_dict["test_data_raw"]["sim_data"].shape[0] == batch_size
    assert test_dict["test_data_raw"]["sim_data"].shape[1] == sim_model.max_time_iter
    if sim_model.is_pde:
        assert test_dict["test_data_raw"]["sim_data"].ndim == 4
        assert test_dict["test_data_raw"]["sim_data"].shape[2] == sim_model.nr
        assert test_dict["test_data_raw"]["sim_data"].shape[3] == sim_model.num_features
    else:
        assert test_dict["test_data_raw"]["sim_data"].ndim == 3
        assert test_dict["test_data_raw"]["sim_data"].shape[2] == sim_model.num_features
    assert test_dict["test_data_process"]["parameters"].ndim == 2
    assert test_dict["test_data_process"]["parameters"].shape[0] == batch_size
    assert (
        test_dict["test_data_process"]["parameters"].shape[1]
        == sim_model.num_hidden_params
    )
    assert test_dict["test_data_process"]["summary_conditions"].shape[0] == batch_size
    assert (
        test_dict["test_data_process"]["summary_conditions"].shape[1]
        == sim_model.max_time_iter
    )
    if sim_model.is_pde:
        assert test_dict["test_data_process"]["summary_conditions"].ndim == 4
        assert (
            test_dict["test_data_process"]["summary_conditions"].shape[2]
            == sim_model.nr
        )
        assert (
            test_dict["test_data_process"]["summary_conditions"].shape[3]
            == sim_model.num_features
        )
    else:
        assert test_dict["test_data_process"]["summary_conditions"].ndim == 3
        assert (
            test_dict["test_data_process"]["summary_conditions"].shape[2]
            == sim_model.num_features
        )
    assert test_dict["posterior_samples"].ndim == 3
    assert test_dict["posterior_samples"].shape[0] == batch_size
    assert test_dict["posterior_samples"].shape[1] == n_samples
    assert test_dict["posterior_samples"].shape[2] == sim_model.num_hidden_params
    assert test_dict["posterior_samples_unnorm"].ndim == 3
    assert test_dict["posterior_samples_unnorm"].shape[0] == batch_size
    assert test_dict["posterior_samples_unnorm"].shape[1] == n_samples
    assert test_dict["posterior_samples_unnorm"].shape[2] == sim_model.num_hidden_params


@pytest.mark.parametrize("model_name", models)
def test_evaluater_evaluate_bayesflow_model_plot_loss(model_name):
    sim_model, dummy_amortizer, dummy_trainer = setup_dummy_objects(model_name)
    num_iterations = random.randint(1, 10)
    dummy_losses = pd.DataFrame(
        np.random.uniform(-100, 100, size=(num_iterations, 1)), columns=["Default.Loss"]
    )

    plot_settings = simulation_settings[model_name]["plot_settings"]
    plot_settings["show_plot"] = True
    plot_settings["show_time"] = 0 if AUTO_CLOSE_PLOTS else None

    dummy_eval_settings = {
        "batch_size": random.randint(1, 8),
        "n_samples": random.randint(2, 8),
        "plot_prior": False,
        "plot_sim_data": False,
        "plot_loss": True,
        "plot_latent": False,
        "plot_sbc_histogram": False,
        "plot_sbc_ecdf": False,
        "plot_true_vs_estimated": False,
        "plot_posterior": False,
        "plot_post_with_prior": False,
        "plot_resimulation": False,
    }

    evaluater = Evaluater(
        sim_model,
        dummy_amortizer,
        dummy_trainer,
        simulation_settings[model_name]["plot_settings"],
        dummy_eval_settings,
    )

    evaluater.load_losses(dummy_losses)
    evaluater.evaluate_bayesflow_model("pytest")

    assert os.path.exists(os.path.join("pytest", "loss.png"))
    if os.path.exists(os.path.join("pytest", "loss.png")):
        os.remove(os.path.join("pytest", "loss.png"))
    if os.path.exists("pytest"):
        os.rmdir("pytest")


@pytest.mark.parametrize("model_name", models)
def test_evaluater_evaluate_bayesflow_model_plot_latent(model_name):
    sim_model, dummy_amortizer, dummy_trainer = setup_dummy_objects(model_name)
    plot_settings = simulation_settings[model_name]["plot_settings"]
    plot_settings["show_plot"] = True
    plot_settings["show_time"] = 0 if AUTO_CLOSE_PLOTS else None
    batch_size = random.randint(1, 8)

    dummy_eval_settings = {
        "batch_size": batch_size,
        "n_samples": random.randint(2, 8),
        "plot_prior": False,
        "plot_sim_data": False,
        "plot_loss": False,
        "plot_latent": True,
        "plot_sbc_histogram": False,
        "plot_sbc_ecdf": False,
        "plot_true_vs_estimated": False,
        "plot_posterior": False,
        "plot_post_with_prior": False,
        "plot_resimulation": False,
    }

    evaluater = Evaluater(
        sim_model,
        dummy_amortizer,
        dummy_trainer,
        simulation_settings[model_name]["plot_settings"],
        dummy_eval_settings,
    )

    evaluater.evaluate_bayesflow_model("pytest")

    assert os.path.exists(os.path.join("pytest", "latent_2d.png"))
    if os.path.exists(os.path.join("pytest", "latent_2d.png")):
        os.remove(os.path.join("pytest", "latent_2d.png"))
    if os.path.exists("pytest"):
        os.rmdir("pytest")


"""
@pytest.mark.parametrize("model_name", models)
def test_evaluater_evaluate_bayesflow_model_plot_sbc_hist(model_name):
    sim_model, dummy_amortizer, dummy_trainer = setup_dummy_objects(model_name)
    plot_settings = simulation_settings[model_name]["plot_settings"]
    plot_settings["show_plot"] = True
    plot_settings["show_time"] = 0 if AUTO_CLOSE_PLOTS else None
    batch_size = random.randint(1, 8)

    dummy_eval_settings = {
        "batch_size": batch_size,
        "n_samples": random.randint(1, 8),
        "plot_prior": False,
        "plot_sim_data": False,
        "plot_loss": False,
        "plot_latent": False,
        "plot_sbc_histogram": True,
        "plot_sbc_ecdf": False,
        "plot_true_vs_estimated": False,
        "plot_posterior": False,
        "plot_post_with_prior": False,
        "plot_resimulation": False,
    }

    evaluater = Evaluater(
        sim_model,
        dummy_amortizer,
        dummy_trainer,
        simulation_settings[model_name]["plot_settings"],
        dummy_eval_settings,
    )

    evaluater.evaluate_bayesflow_model("pytest")

    assert os.path.exists(os.path.join("pytest", "sbc_hist.png"))
    if os.path.exists(os.path.join("pytest", "sbc_hist.png")):
        os.remove(os.path.join("pytest", "sbc_hist.png"))
    if os.path.exists("pytest"):
        os.rmdir("pytest")
"""


@pytest.mark.parametrize("model_name", models)
def test_evaluater_evaluate_bayesflow_model_plot_sbc_ecdf(model_name):
    sim_model, dummy_amortizer, dummy_trainer = setup_dummy_objects(model_name)
    plot_settings = simulation_settings[model_name]["plot_settings"]
    plot_settings["show_plot"] = True
    plot_settings["show_time"] = 0 if AUTO_CLOSE_PLOTS else None

    dummy_eval_settings = {
        "batch_size": random.randint(1, 8),
        "n_samples": random.randint(2, 8),
        "plot_prior": False,
        "plot_sim_data": False,
        "plot_loss": False,
        "plot_latent": False,
        "plot_sbc_histogram": False,
        "plot_sbc_ecdf": True,
        "plot_true_vs_estimated": False,
        "plot_posterior": False,
        "plot_post_with_prior": False,
        "plot_resimulation": False,
    }

    evaluater = Evaluater(
        sim_model,
        dummy_amortizer,
        dummy_trainer,
        simulation_settings[model_name]["plot_settings"],
        dummy_eval_settings,
    )

    evaluater.evaluate_bayesflow_model("pytest")

    assert os.path.exists(os.path.join("pytest", "sbc_ecdf.png"))
    if os.path.exists(os.path.join("pytest", "sbc_ecdf.png")):
        os.remove(os.path.join("pytest", "sbc_ecdf.png"))
    if os.path.exists("pytest"):
        os.rmdir("pytest")


@pytest.mark.parametrize("model_name", models)
def test_evaluater_evaluate_bayesflow_model_plot_true_vs_estimated(model_name):
    sim_model, dummy_amortizer, dummy_trainer = setup_dummy_objects(model_name)
    plot_settings = simulation_settings[model_name]["plot_settings"]
    plot_settings["show_plot"] = True
    plot_settings["show_time"] = 0 if AUTO_CLOSE_PLOTS else None

    dummy_eval_settings = {
        "batch_size": random.randint(2, 8),
        "n_samples": random.randint(2, 8),
        "plot_prior": False,
        "plot_sim_data": False,
        "plot_loss": False,
        "plot_latent": False,
        "plot_sbc_histogram": False,
        "plot_sbc_ecdf": False,
        "plot_true_vs_estimated": True,
        "plot_posterior": False,
        "plot_post_with_prior": False,
        "plot_resimulation": False,
    }

    evaluater = Evaluater(
        sim_model,
        dummy_amortizer,
        dummy_trainer,
        simulation_settings[model_name]["plot_settings"],
        dummy_eval_settings,
    )

    evaluater.evaluate_bayesflow_model("pytest")

    assert os.path.exists(os.path.join("pytest", "true_vs_estimated.png"))
    if os.path.exists(os.path.join("pytest", "true_vs_estimated.png")):
        os.remove(os.path.join("pytest", "true_vs_estimated.png"))
    if os.path.exists("pytest"):
        os.rmdir("pytest")


@pytest.mark.parametrize("model_name", models)
def test_evaluater_evaluate_bayesflow_model_plot_posterior(model_name):
    sim_model, dummy_amortizer, dummy_trainer = setup_dummy_objects(model_name)
    plot_settings = simulation_settings[model_name]["plot_settings"]
    plot_settings["show_plot"] = True
    plot_settings["show_time"] = 0 if AUTO_CLOSE_PLOTS else None

    dummy_eval_settings = {
        "batch_size": random.randint(2, 8),
        "n_samples": random.randint(2, 8),
        "plot_prior": False,
        "plot_sim_data": False,
        "plot_loss": False,
        "plot_latent": False,
        "plot_sbc_histogram": False,
        "plot_sbc_ecdf": False,
        "plot_true_vs_estimated": False,
        "plot_posterior": True,
        "plot_post_with_prior": False,
        "plot_resimulation": False,
    }

    evaluater = Evaluater(
        sim_model,
        dummy_amortizer,
        dummy_trainer,
        simulation_settings[model_name]["plot_settings"],
        dummy_eval_settings,
    )

    evaluater.evaluate_bayesflow_model("pytest")

    assert os.path.exists(os.path.join("pytest", "posterior.png"))
    if os.path.exists(os.path.join("pytest", "posterior.png")):
        os.remove(os.path.join("pytest", "posterior.png"))
    if os.path.exists("pytest"):
        os.rmdir("pytest")


@pytest.mark.parametrize("model_name", models)
def test_evaluater_evaluate_bayesflow_model_plot_posterior_with_prior(model_name):
    sim_model, dummy_amortizer, dummy_trainer = setup_dummy_objects(model_name)
    plot_settings = simulation_settings[model_name]["plot_settings"]
    plot_settings["show_plot"] = True
    plot_settings["show_time"] = 0 if AUTO_CLOSE_PLOTS else None

    dummy_eval_settings = {
        "batch_size": random.randint(2, 8),
        "n_samples": random.randint(2, 8),
        "plot_prior": False,
        "plot_sim_data": False,
        "plot_loss": False,
        "plot_latent": False,
        "plot_sbc_histogram": False,
        "plot_sbc_ecdf": False,
        "plot_true_vs_estimated": False,
        "plot_posterior": False,
        "plot_post_with_prior": True,
        "plot_resimulation": False,
    }

    evaluater = Evaluater(
        sim_model,
        dummy_amortizer,
        dummy_trainer,
        simulation_settings[model_name]["plot_settings"],
        dummy_eval_settings,
    )

    evaluater.evaluate_bayesflow_model("pytest")

    assert os.path.join(os.path.join("pytest", "compare_prior_post.png"))
    if os.path.exists(os.path.join("pytest", "compare_prior_post.png")):
        os.remove(os.path.join("pytest", "compare_prior_post.png"))
    if os.path.exists("pytest"):
        os.rmdir("pytest")


@pytest.mark.parametrize("model_name", models)
def test_evaluater_evaluate_bayesflow_model_plot_resimulation(model_name):
    sim_model, dummy_amortizer, dummy_trainer = setup_dummy_objects(model_name)
    plot_settings = simulation_settings[model_name]["plot_settings"]
    plot_settings["show_plot"] = True
    plot_settings["show_time"] = 0 if AUTO_CLOSE_PLOTS else None
    plot_settings["num_plots"] = 4

    dummy_eval_settings = {
        "batch_size": random.randint(4, 8),
        "n_samples": random.randint(2, 8),
        "plot_prior": False,
        "plot_sim_data": False,
        "plot_loss": False,
        "plot_latent": False,
        "plot_sbc_histogram": False,
        "plot_sbc_ecdf": False,
        "plot_true_vs_estimated": False,
        "plot_posterior": False,
        "plot_post_with_prior": False,
        "plot_resimulation": True,
    }

    evaluater = Evaluater(
        sim_model,
        dummy_amortizer,
        dummy_trainer,
        plot_settings,
        dummy_eval_settings,
    )

    evaluater.evaluate_bayesflow_model("pytest")

    assert os.path.exists(os.path.join("pytest", "resimulation.png"))
    if os.path.exists(os.path.join("pytest", "resimulation.png")):
        os.remove(os.path.join("pytest", "resimulation.png"))
    if os.path.exists("pytest"):
        os.rmdir("pytest")
