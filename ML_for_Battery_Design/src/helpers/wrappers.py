import os

from ML_for_Battery_Design.src.helpers.constants import inference_settings
from ML_for_Battery_Design.src.helpers.initializer import Initializer


def train_online(**kwargs: str) -> None:
    """Wrapper for executing online training based on user input"""
    if not bool(kwargs["--skip_wrappers"]):
        initializer = Initializer(**kwargs)
        train_settings = inference_settings[initializer.sim_model_name]["training"]
        if kwargs["--test_mode"]:
            train_settings["num_epochs"] = 1
            train_settings["it_per_epoch"] = 1
            train_settings["batch_size"] = 4
        losses = initializer.trainer.train_online(
            epochs=train_settings["num_epochs"],
            iterations_per_epoch=train_settings["it_per_epoch"],
            batch_size=train_settings["batch_size"],
        )
        initializer.save_losses(losses)
        initializer.evaluater.load_losses(losses)
        initializer.evaluater.evaluate_sim_model(initializer.file_manager("result"))
        initializer.evaluater.evaluate_bayesflow_model(
            initializer.file_manager("result")
        )


def train_offline(**kwargs: str) -> None:
    """Wrapper for executing offline training based on user input"""
    if not bool(kwargs["--skip_wrappers"]):
        initializer = Initializer(**kwargs)
        train_settings = inference_settings[initializer.sim_model_name]["training"]
        if kwargs["--test_mode"]:
            train_settings["num_epochs"] = 1
            train_settings["it_per_epoch"] = 1
            train_settings["batch_size"] = 4
        train_settings = inference_settings[initializer.sim_model_name]["training"]
        data_dict = initializer.load_hdf5_data()
        losses = initializer.trainer.train_offline(
            data_dict,
            epochs=train_settings["num_epochs"],
            it_per_epoch=train_settings["it_per_epoch"],
            batch_size=train_settings["batch_size"],
        )
        initializer.save_losses(losses)
        initializer.evaluater.load_losses(losses)
        initializer.evaluater.evaluate_sim_model(initializer.file_manager("result"))
        initializer.evaluater.evaluate_bayesflow_model(
            initializer.file_manager("result")
        )


def generate_data(**kwargs: str) -> None:
    """Wrapper for generating simulation data based on user input"""
    if not bool(kwargs["--skip_wrappers"]):
        initializer = Initializer(**kwargs)
        data_settings = inference_settings[initializer.sim_model_name]["generate_data"]
        if kwargs["--test_mode"]:
            data_settings["total_n_sim"] = 8
            data_settings["chunk_size"] = 4
        initializer.generate_hdf5_data()


def analyze_sim(**kwargs: str) -> None:
    """Wrapper for analyzing simulation model prior and simulation data based on user input"""
    if not bool(kwargs["--skip_wrappers"]):
        initializer = Initializer(**kwargs)
        initializer.evaluater.evaluate_sim_model(initializer.file_manager("result"))


def evaluate(**kwargs: str) -> None:
    """Wrapper for evaluating a trained model based on user input"""
    if not bool(kwargs["--skip_wrappers"]):
        initializer = Initializer(**kwargs)
        initializer.trainer.load_pretrained_network()
        initializer.evaluater.load_losses(
            os.path.join(initializer.file_manager("results"), "losses.pickle")
        )
        initializer.evaluater.evaluate_bayesflow_model()
