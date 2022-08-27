import pytest
from bayesflow.trainers import Trainer

import ML_for_Battery_Design.src.main as main
from ML_for_Battery_Design.src.helpers.constants import sim_model_collection
from ML_for_Battery_Design.src.helpers.filemanager import FileManager
from ML_for_Battery_Design.src.helpers.initializer import Initializer
from ML_for_Battery_Design.src.simulation.simulation_model import SimulationModel

models = ["linear_ode_system"]

modes = ["train_online", "train_offline", "generate_data", "analyze_sim", "evaluate"]


def setup_user_args(mode, sim_model, save_model=False):
    if mode == "train_online":
        user_input = [mode, sim_model, "FC", "pytest_file"]
        if save_model:
            user_input += ["--save_model"]
    elif mode == "train_offline":
        user_input = [mode, sim_model, "pytest_data", "FC", "pytest_file"]
        if save_model:
            user_input += ["--save_model"]
    elif mode == "generate_data":
        user_input = [mode, sim_model, "pytest_data"]
    elif mode == "analyze_sim":
        user_input = [mode, sim_model, "pytest_file"]
    elif mode == "evaluate":
        user_input = [mode, sim_model, "pytest_data", "pytest_file"]
    else:
        raise ValueError("{} is not a valid mode".format(mode))

    args = main.main(user_input)

    return args


@pytest.mark.parametrize("model_name", models)
def test_initializer_init_train_online(model_name):
    args = setup_user_args("train_online", model_name)

    initializer = Initializer(**args)

    assert isinstance(initializer.sim_model_name, str)
    assert isinstance(initializer.summary_net_name, str)
    assert isinstance(initializer.data_name, str)
    assert isinstance(initializer.filename, str)
    assert isinstance(initializer.save_model, bool)
    assert isinstance(initializer.sim_model, SimulationModel)
    assert isinstance(initializer.sim_model, sim_model_collection[model_name])
    assert isinstance(initializer.mode, str)
    assert isinstance(initializer.file_manager, FileManager)
    assert initializer.sim_model_name == model_name
    assert initializer.summary_net_name == "FC"
    assert initializer.data_name == "online"
    assert initializer.filename == "pytest_file"
    assert not initializer.save_model
    assert initializer.mode == "train_online"
    assert isinstance(initializer.trainer, Trainer)


@pytest.mark.parametrize("model_name", models)
def test_initializer_init_train_offline(model_name):
    args = setup_user_args("train_offline", model_name)

    initializer = Initializer(**args)

    assert isinstance(initializer.sim_model_name, str)
    assert isinstance(initializer.summary_net_name, str)
    assert isinstance(initializer.data_name, str)
    assert isinstance(initializer.filename, str)
    assert isinstance(initializer.save_model, bool)
    assert isinstance(initializer.sim_model, SimulationModel)
    assert isinstance(initializer.sim_model, sim_model_collection[model_name])
    assert isinstance(initializer.mode, str)
    assert isinstance(initializer.file_manager, FileManager)
    assert initializer.sim_model_name == model_name
    assert initializer.summary_net_name == "FC"
    assert initializer.data_name == "pytest_data"
    assert initializer.filename == "pytest_file"
    assert not initializer.save_model
    assert initializer.mode == "train_offline"
    assert isinstance(initializer.trainer, Trainer)


@pytest.mark.parametrize("model_name", models)
def test_initializer_init_generate_data(model_name):
    args = setup_user_args("generate_data", model_name)

    initializer = Initializer(**args)

    assert isinstance(initializer.sim_model_name, str)
    assert isinstance(initializer.data_name, str)
    assert isinstance(initializer.save_model, bool)
    assert isinstance(initializer.sim_model, SimulationModel)
    assert isinstance(initializer.sim_model, sim_model_collection[model_name])
    assert isinstance(initializer.mode, str)
    assert isinstance(initializer.file_manager, FileManager)
    assert initializer.sim_model_name == model_name
    assert initializer.summary_net_name is None
    assert initializer.data_name == "pytest_data"
    assert initializer.filename is None
    assert not initializer.save_model
    assert initializer.mode == "generate_data"


@pytest.mark.parametrize("model_name", models)
def test_initializer_init_analyze_sim(model_name):
    args = setup_user_args("analyze_sim", model_name)

    initializer = Initializer(**args)

    assert isinstance(initializer.sim_model_name, str)
    assert isinstance(initializer.filename, str)
    assert isinstance(initializer.save_model, bool)
    assert isinstance(initializer.sim_model, SimulationModel)
    assert isinstance(initializer.sim_model, sim_model_collection[model_name])
    assert isinstance(initializer.mode, str)
    assert isinstance(initializer.file_manager, FileManager)
    assert initializer.sim_model_name == model_name
    assert initializer.summary_net_name is None
    assert initializer.data_name is None
    assert initializer.filename == "pytest_file"
    assert not initializer.save_model
    assert initializer.mode == "analyze_sim"


@pytest.mark.parametrize("model_name", models)
def test_initializer_init_evaluate(model_name):
    args = setup_user_args("evaluate", model_name)

    initializer = Initializer(**args)

    assert isinstance(initializer.sim_model_name, str)
    assert isinstance(initializer.data_name, str)
    assert isinstance(initializer.filename, str)
    assert isinstance(initializer.save_model, bool)
    assert isinstance(initializer.sim_model, SimulationModel)
    assert isinstance(initializer.sim_model, sim_model_collection[model_name])
    assert isinstance(initializer.mode, str)
    assert isinstance(initializer.file_manager, FileManager)
    assert initializer.sim_model_name == model_name
    assert initializer.summary_net_name is None
    assert initializer.data_name == "pytest_data"
    assert initializer.filename == "pytest_file"
    assert not initializer.save_model
    assert initializer.mode == "evaluate"
