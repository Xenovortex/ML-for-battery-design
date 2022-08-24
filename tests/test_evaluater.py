import os
import random

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


def setup_dummy_objects(model_name):
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
    configurator = Processing(
        inference_settings[model_name]["processing"],
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
