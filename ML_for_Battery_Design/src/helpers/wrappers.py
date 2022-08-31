import os
import pathlib

from ML_for_Battery_Design.src.helpers.constants import inference_settings
from ML_for_Battery_Design.src.helpers.initializer import Initializer


def train_online(**kwargs: str) -> None:
    """Wrapper for executing online training based on user input"""
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
    pathlib.Path(initializer.file_manager("result")).mkdir(parents=True, exist_ok=True)
    losses.to_pickle(os.path.join(initializer.file_manager("result"), "losses.pickle"))
    initializer.evaluater.load_losses(losses)
    initializer.evaluater.evaluate_sim_model(initializer.file_manager("result"))
    initializer.evaluater.evaluate_bayesflow_model(initializer.file_manager("result"))


def train_offline(**kwargs: str) -> None:
    """Wrapper for executing offline training based on user input"""
    pass


def generate_data(**kwargs: str) -> None:
    """Wrapper for generating simulatoin data based on user input"""
    pass


def analyze_sim(**kwargs: str) -> None:
    """Wrapper for analyzing simulation model prior and simulation data based on user input"""
    pass


def evaluate(**kwargs: str) -> None:
    """Wrapper for evaluating a trained model based on user input"""
    pass
