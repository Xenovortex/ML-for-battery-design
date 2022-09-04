import os
import random

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from bayesflow.amortized_inference import AmortizedPosterior
from bayesflow.networks import InvertibleNetwork
from bayesflow.trainers import Trainer

from ML_for_Battery_Design.src.helpers.constants import (
    architecture_settings,
    inference_settings,
    sim_model_collection,
    simulation_settings,
)
from ML_for_Battery_Design.src.helpers.evaluater import Evaluater
from ML_for_Battery_Design.src.helpers.filemanager import FileManager
from ML_for_Battery_Design.src.helpers.initializer import Initializer
from ML_for_Battery_Design.src.helpers.processing import Processing
from ML_for_Battery_Design.src.helpers.summary import FC_Network
from ML_for_Battery_Design.src.simulation.simulation_model import SimulationModel
from tests.constants import models, modes
from tests.helpers import setup_user_args


@pytest.mark.parametrize("model_name", models)
def test_initializer_init_train_online(model_name):
    args = setup_user_args("train_online", model_name)

    initializer = Initializer(
        architecture_settings, inference_settings, simulation_settings, **args
    )

    assert isinstance(initializer.sim_model_name, str)
    assert isinstance(initializer.summary_net_name, str)
    assert isinstance(initializer.data_name, str)
    assert isinstance(initializer.filename, str)
    assert isinstance(initializer.save_model, bool)
    assert isinstance(initializer.test_mode, bool)
    assert isinstance(initializer.sim_model, SimulationModel)
    assert isinstance(initializer.sim_model, sim_model_collection[model_name])
    assert isinstance(initializer.mode, str)
    assert isinstance(initializer.file_manager, FileManager)
    assert isinstance(initializer.trainer, Trainer)
    assert isinstance(initializer.evaluater, Evaluater)
    assert initializer.sim_model_name == model_name
    assert initializer.summary_net_name == "FC"
    assert initializer.data_name == "online"
    assert initializer.filename == "pytest_file"
    assert not initializer.save_model
    assert initializer.test_mode
    assert initializer.mode == "train_online"


@pytest.mark.parametrize("model_name", models)
def test_initializer_init_train_offline(model_name):
    args = setup_user_args("train_offline", model_name)

    initializer = Initializer(
        architecture_settings, inference_settings, simulation_settings, **args
    )

    assert isinstance(initializer.sim_model_name, str)
    assert isinstance(initializer.summary_net_name, str)
    assert isinstance(initializer.data_name, str)
    assert isinstance(initializer.filename, str)
    assert isinstance(initializer.save_model, bool)
    assert isinstance(initializer.test_mode, bool)
    assert isinstance(initializer.sim_model, SimulationModel)
    assert isinstance(initializer.sim_model, sim_model_collection[model_name])
    assert isinstance(initializer.mode, str)
    assert isinstance(initializer.file_manager, FileManager)
    assert isinstance(initializer.trainer, Trainer)
    assert isinstance(initializer.evaluater, Evaluater)
    assert initializer.sim_model_name == model_name
    assert initializer.summary_net_name == "FC"
    assert initializer.data_name == "pytest_data"
    assert initializer.filename == "pytest_file"
    assert not initializer.save_model
    assert initializer.test_mode
    assert initializer.mode == "train_offline"


@pytest.mark.parametrize("model_name", models)
def test_initializer_init_generate_data(model_name):
    args = setup_user_args("generate_data", model_name)

    initializer = Initializer(
        architecture_settings, inference_settings, simulation_settings, **args
    )

    assert isinstance(initializer.sim_model_name, str)
    assert isinstance(initializer.data_name, str)
    assert isinstance(initializer.save_model, bool)
    assert isinstance(initializer.test_mode, bool)
    assert isinstance(initializer.sim_model, SimulationModel)
    assert isinstance(initializer.sim_model, sim_model_collection[model_name])
    assert isinstance(initializer.mode, str)
    assert isinstance(initializer.file_manager, FileManager)
    assert initializer.sim_model_name == model_name
    assert initializer.summary_net_name is None
    assert initializer.data_name == "pytest_data"
    assert initializer.filename is None
    assert not initializer.save_model
    assert initializer.test_mode
    assert initializer.mode == "generate_data"


@pytest.mark.parametrize("model_name", models)
def test_initializer_init_analyze_sim(model_name):
    args = setup_user_args("analyze_sim", model_name)

    initializer = Initializer(
        architecture_settings, inference_settings, simulation_settings, **args
    )

    assert isinstance(initializer.sim_model_name, str)
    assert isinstance(initializer.save_model, bool)
    assert isinstance(initializer.test_mode, bool)
    assert isinstance(initializer.sim_model, SimulationModel)
    assert isinstance(initializer.sim_model, sim_model_collection[model_name])
    assert isinstance(initializer.mode, str)
    assert isinstance(initializer.file_manager, FileManager)
    assert isinstance(initializer.evaluater, Evaluater)
    assert initializer.sim_model_name == model_name
    assert initializer.summary_net_name is None
    assert initializer.data_name is None
    assert initializer.filename is None
    assert not initializer.save_model
    assert initializer.test_mode
    assert initializer.mode == "analyze_sim"


@pytest.mark.parametrize("model_name", models)
def test_initializer_init_evaluate(model_name):
    args = setup_user_args("train_offline", model_name)
    initializer = Initializer(
        architecture_settings, inference_settings, simulation_settings, **args
    )
    initializer.save_setup()
    args = setup_user_args("evaluate", model_name)
    initializer = Initializer(None, None, None, **args)

    assert isinstance(initializer.sim_model_name, str)
    assert isinstance(initializer.summary_net_name, str)
    assert isinstance(initializer.data_name, str)
    assert isinstance(initializer.filename, str)
    assert isinstance(initializer.save_model, bool)
    assert isinstance(initializer.test_mode, bool)
    assert isinstance(initializer.sim_model, SimulationModel)
    assert isinstance(initializer.sim_model, sim_model_collection[model_name])
    assert isinstance(initializer.mode, str)
    assert isinstance(initializer.file_manager, FileManager)
    assert isinstance(initializer.trainer, Trainer)
    assert isinstance(initializer.evaluater, Evaluater)
    assert initializer.sim_model_name == model_name
    assert initializer.summary_net_name == "FC"
    assert initializer.data_name == "pytest_data"
    assert initializer.filename == "pytest_file"
    assert not initializer.save_model
    assert initializer.test_mode
    assert initializer.mode == "evaluate"

    if os.path.exists(
        os.path.join("models", model_name, "pytest_data", "pytest_file", "setup.pickle")
    ):
        os.remove(
            os.path.join(
                "models", model_name, "pytest_data", "pytest_file", "setup.pickle"
            )
        )
    if os.path.exists(os.path.join("models", model_name, "pytest_data", "pytest_file")):
        os.rmdir(os.path.join("models", model_name, "pytest_data", "pytest_file"))
    if os.path.exists(os.path.join("models", model_name, "pytest_data")):
        os.rmdir(os.path.join("models", model_name, "pytest_data"))


@pytest.mark.parametrize("mode", modes)
@pytest.mark.parametrize("model_name", models)
def test_initializer_init_architecture_error(mode, model_name, capsys):
    args = setup_user_args(mode, model_name)
    with pytest.raises(ValueError):
        initializer = Initializer({}, {}, {}, **args)
        out, err = capsys.readouterr()
        assert out == ""
        assert (
            err
            == "{} - init: simulation model {} has no architecture settings".format(
                initializer.__class__.__name__, model_name
            )
        )


@pytest.mark.parametrize("mode", modes)
@pytest.mark.parametrize("model_name", models)
def test_initializer_init_inference_error(mode, model_name, capsys):
    args = setup_user_args(mode, model_name)
    with pytest.raises(ValueError):
        initializer = Initializer(architecture_settings, {}, {}, **args)
        out, err = capsys.readouterr()
        assert out == ""
        assert err == "{} - init: simulation model {} has no inference settings".format(
            initializer.__class__.__name__, model_name
        )


@pytest.mark.parametrize("mode", modes)
@pytest.mark.parametrize("model_name", models)
def test_initializer_init_simulation_error(mode, model_name, capsys):
    args = setup_user_args(mode, model_name)
    with pytest.raises(ValueError):
        initializer = Initializer(architecture_settings, inference_settings, {}, **args)
        out, err = capsys.readouterr()
        assert out == ""
        assert (
            err
            == "{} - init: simulation model {} has no simulation settings".format(
                initializer.__class__.__name__, model_name
            )
        )


@pytest.mark.parametrize("mode", modes)
@pytest.mark.parametrize("model_name", models)
def test_initializer_get_sim_model(mode, model_name):
    args = setup_user_args(mode, model_name)
    initializer = Initializer(
        architecture_settings, inference_settings, simulation_settings, **args
    )
    if mode == "train_offline":
        initializer.save_setup()
    sim_model = initializer.get_sim_model()

    assert isinstance(sim_model, SimulationModel)
    assert isinstance(sim_model, sim_model_collection[model_name])
    if mode == "evaluate":
        if os.path.exists(
            os.path.join(
                "models", model_name, "pytest_data", "pytest_file", "setup.pickle"
            )
        ):
            os.remove(
                os.path.join(
                    "models", model_name, "pytest_data", "pytest_file", "setup.pickle"
                )
            )
        if os.path.exists(
            os.path.join("models", model_name, "pytest_data", "pytest_file")
        ):
            os.rmdir(os.path.join("models", model_name, "pytest_data", "pytest_file"))
        if os.path.exists(os.path.join("models", model_name, "pytest_data")):
            os.rmdir(os.path.join("models", model_name, "pytest_data"))


@pytest.mark.parametrize("mode", modes)
def test_initializer_get_sim_model_sim_model_error(mode, capsys):
    if mode == "evaluate":
        args = setup_user_args("train_offline", random.choice(models))
        initializer = Initializer(
            architecture_settings, inference_settings, simulation_settings, **args
        )
        initializer.sim_model_name = "pytest"
        initializer.update_filemanager()
        initializer.save_setup()

    args = setup_user_args(mode, "pytest")

    with pytest.raises(ValueError):
        initializer = Initializer(
            architecture_settings, inference_settings, simulation_settings, **args
        )
        out, err = capsys.readouterr()
        assert out == ""
        assert err == "{} - get_sim_model: {} is not a valid simulation model".format(
            initializer.__class__.__name__, "pytest"
        )

    if mode == "evaluate":
        if os.path.exists(
            os.path.join(
                "models", "pytest", "pytest_data", "pytest_file", "setup.pickle"
            )
        ):
            os.remove(
                os.path.join(
                    "models", "pytest", "pytest_data", "pytest_file", "setup.pickle"
                )
            )
        if os.path.exists(
            os.path.join("models", "pytest", "pytest_data", "pytest_file")
        ):
            os.rmdir(os.path.join("models", "pytest", "pytest_data", "pytest_file"))
        if os.path.exists(os.path.join("models", "pytest", "pytest_data")):
            os.rmdir(os.path.join("models", "pytest", "pytest_data"))


@pytest.mark.parametrize("mode", modes)
@pytest.mark.parametrize("model_name", models)
def test_initializer_get_summary_net(mode, model_name, capsys):
    args = setup_user_args(mode, model_name)
    initializer = Initializer(
        architecture_settings, inference_settings, simulation_settings, **args
    )
    if mode == "train_offline":
        initializer.save_setup()

    if mode in ["train_online", "train_offline", "evaluate"]:
        summary_net = initializer.get_summary_net()
        assert isinstance(summary_net, tf.keras.Model)
        assert isinstance(summary_net, FC_Network)
    else:
        with pytest.raises(ValueError):
            initializer.get_summary_net()
            out, err = capsys.readouterr()
            assert out == ""
            assert (
                err
                == "{} - get_summary_net: {} not found in architecture settings for {} simulation model".format(
                    initializer.__class__.__name__, None, model_name
                )
            )

    if mode == "evaluate":
        if os.path.exists(
            os.path.join(
                "models", model_name, "pytest_data", "pytest_file", "setup.pickle"
            )
        ):
            os.remove(
                os.path.join(
                    "models", model_name, "pytest_data", "pytest_file", "setup.pickle"
                )
            )
        if os.path.exists(
            os.path.join("models", model_name, "pytest_data", "pytest_file")
        ):
            os.rmdir(os.path.join("models", model_name, "pytest_data", "pytest_file"))
        if os.path.exists(os.path.join("models", model_name, "pytest_data")):
            os.rmdir(os.path.join("models", model_name, "pytest_data"))


@pytest.mark.parametrize("mode", modes)
@pytest.mark.parametrize("model_name", models)
def test_initializer_get_summary_net_invalid_sum_net(mode, model_name, capsys):
    args = setup_user_args(mode, model_name)
    if mode != "evaluate":
        initializer = Initializer(
            architecture_settings, inference_settings, simulation_settings, **args
        )
        initializer.summary_net_name = "pytest"

    if mode == "train_offline":
        initializer.update_filemanager()
        initializer.save_setup()

    if mode in ["train_online", "train_offline"]:
        initializer.summary_net_name = "pytest"
        expected_summary_name = "pytest"
    if mode == "evaluate":
        expected_summary_name = "pytest"
    else:
        expected_summary_name = None

    with pytest.raises(ValueError):
        if mode == "evaluate":
            initializer = Initializer(
                architecture_settings, inference_settings, simulation_settings, **args
            )
        else:
            initializer.get_summary_net()
        out, err = capsys.readouterr()

        assert out == ""
        assert (
            err
            == "{} - get_summary_net: {} is not a valid summary network architecture".format(
                initializer.__class__.__name__, expected_summary_name
            )
        )
    if mode == "evaluate":
        if os.path.exists(
            os.path.join(
                "models", model_name, "pytest_data", "pytest_file", "setup.pickle"
            )
        ):
            os.remove(
                os.path.join(
                    "models", model_name, "pytest_data", "pytest_file", "setup.pickle"
                )
            )
        if os.path.exists(
            os.path.join("models", model_name, "pytest_data", "pytest_file")
        ):
            os.rmdir(os.path.join("models", model_name, "pytest_data", "pytest_file"))
        if os.path.exists(os.path.join("models", model_name, "pytest_data")):
            os.rmdir(os.path.join("models", model_name, "pytest_data"))


@pytest.mark.parametrize("mode", modes)
@pytest.mark.parametrize("model_name", models)
def test_initializer_get_inference_net(mode, model_name):
    args = setup_user_args(mode, model_name)
    initializer = Initializer(
        architecture_settings, inference_settings, simulation_settings, **args
    )
    if mode == "train_offline":
        initializer.save_setup()
    inference_net = initializer.get_inference_net()

    assert isinstance(inference_net, InvertibleNetwork)

    if mode == "evaluate":
        if os.path.exists(
            os.path.join(
                "models", model_name, "pytest_data", "pytest_file", "setup.pickle"
            )
        ):
            os.remove(
                os.path.join(
                    "models", model_name, "pytest_data", "pytest_file", "setup.pickle"
                )
            )
        if os.path.exists(
            os.path.join("models", model_name, "pytest_data", "pytest_file")
        ):
            os.rmdir(os.path.join("models", model_name, "pytest_data", "pytest_file"))
        if os.path.exists(os.path.join("models", model_name, "pytest_data")):
            os.rmdir(os.path.join("models", model_name, "pytest_data"))


@pytest.mark.parametrize("mode", modes)
@pytest.mark.parametrize("model_name", models)
def test_initializer_get_inference_net_no_INN_architecture(mode, model_name, capsys):
    args = setup_user_args(mode, model_name)
    initializer = Initializer(
        architecture_settings, inference_settings, simulation_settings, **args
    )
    if mode == "train_offline":
        initializer.save_setup()

    initializer.architecture = {}

    with pytest.raises(ValueError):
        initializer.get_inference_net()
        out, err = capsys.readouterr()

        assert out == ""
        assert (
            err
            == "{} - get_inference_net: INN not found in architecture settings for {} simulation model".format(
                initializer.__class__.__name__, "pytest"
            )
        )
    if mode == "evaluate":
        if os.path.exists(
            os.path.join(
                "models", model_name, "pytest_data", "pytest_file", "setup.pickle"
            )
        ):
            os.remove(
                os.path.join(
                    "models", model_name, "pytest_data", "pytest_file", "setup.pickle"
                )
            )
        if os.path.exists(
            os.path.join("models", model_name, "pytest_data", "pytest_file")
        ):
            os.rmdir(os.path.join("models", model_name, "pytest_data", "pytest_file"))
        if os.path.exists(os.path.join("models", model_name, "pytest_data")):
            os.rmdir(os.path.join("models", model_name, "pytest_data"))


@pytest.mark.parametrize("mode", modes)
@pytest.mark.parametrize("model_name", models)
def test_initializer_get_amortizer(mode, model_name, capsys):
    args = setup_user_args(mode, model_name)
    initializer = Initializer(
        architecture_settings, inference_settings, simulation_settings, **args
    )
    if mode == "train_offline":
        initializer.save_setup()

    if mode in ["train_online", "train_offline", "evaluate"]:
        amortizer = initializer.get_amortizer()
        assert isinstance(amortizer, AmortizedPosterior)
    else:
        with pytest.raises(ValueError):
            initializer.get_amortizer()
            out, err = capsys.readouterr()

            assert out == ""
            assert (
                err
                == "{} - get_summary_net: {} not found in architecture settings for {} simulation model".format(
                    initializer.__class__.__name__, None, model_name
                )
            )

    if mode == "evaluate":
        if os.path.exists(
            os.path.join(
                "models", model_name, "pytest_data", "pytest_file", "setup.pickle"
            )
        ):
            os.remove(
                os.path.join(
                    "models", model_name, "pytest_data", "pytest_file", "setup.pickle"
                )
            )
        if os.path.exists(
            os.path.join("models", model_name, "pytest_data", "pytest_file")
        ):
            os.rmdir(os.path.join("models", model_name, "pytest_data", "pytest_file"))
        if os.path.exists(os.path.join("models", model_name, "pytest_data")):
            os.rmdir(os.path.join("models", model_name, "pytest_data"))


@pytest.mark.parametrize("mode", modes)
@pytest.mark.parametrize("model_name", models)
def test_initializer_get_configurator(mode, model_name):
    args = setup_user_args(mode, model_name)
    initializer = Initializer(
        architecture_settings, inference_settings, simulation_settings, **args
    )
    if mode == "train_offline":
        initializer.save_setup()
    configurator = initializer.get_configurator()

    assert isinstance(configurator, Processing)

    if mode == "evaluate":
        if os.path.exists(
            os.path.join(
                "models", model_name, "pytest_data", "pytest_file", "setup.pickle"
            )
        ):
            os.remove(
                os.path.join(
                    "models", model_name, "pytest_data", "pytest_file", "setup.pickle"
                )
            )
        if os.path.exists(
            os.path.join("models", model_name, "pytest_data", "pytest_file")
        ):
            os.rmdir(os.path.join("models", model_name, "pytest_data", "pytest_file"))
        if os.path.exists(os.path.join("models", model_name, "pytest_data")):
            os.rmdir(os.path.join("models", model_name, "pytest_data"))


@pytest.mark.parametrize("mode", modes)
@pytest.mark.parametrize("model_name", models)
def test_initializer_get_configurator_no_inference_settings(mode, model_name, capsys):
    args = setup_user_args(mode, model_name)
    initializer = Initializer(
        architecture_settings, inference_settings, simulation_settings, **args
    )
    if mode == "train_offline":
        initializer.save_setup()

    initializer.inference = {}

    with pytest.raises(ValueError):
        initializer.get_configurator()
        out, err = capsys.readouterr()

        assert out == ""
        assert (
            err
            == "{} - get_configurator: processing not found in inference settings for {} simulation model".format(
                initializer.__class__.__name__, "pytest"
            )
        )

    if mode == "evaluate":
        if os.path.exists(
            os.path.join(
                "models", model_name, "pytest_data", "pytest_file", "setup.pickle"
            )
        ):
            os.remove(
                os.path.join(
                    "models", model_name, "pytest_data", "pytest_file", "setup.pickle"
                )
            )
        if os.path.exists(
            os.path.join("models", model_name, "pytest_data", "pytest_file")
        ):
            os.rmdir(os.path.join("models", model_name, "pytest_data", "pytest_file"))
        if os.path.exists(os.path.join("models", model_name, "pytest_data")):
            os.rmdir(os.path.join("models", model_name, "pytest_data"))


@pytest.mark.parametrize("mode", modes)
@pytest.mark.parametrize("model_name", models)
def test_initilizer_get_trainer(mode, model_name, capsys):
    args = setup_user_args(mode, model_name)
    initializer = Initializer(
        architecture_settings, inference_settings, simulation_settings, **args
    )
    if mode == "train_offline":
        initializer.save_setup()

    if mode in ["train_online", "train_offline", "evaluate"]:
        trainer = initializer.get_trainer()
        assert isinstance(trainer, Trainer)
    else:
        with pytest.raises(ValueError):
            initializer.get_summary_net()
            out, err = capsys.readouterr()
            assert out == ""
            assert (
                err
                == "{} - get_summary_net: {} not found in architecture settings for {} simulation model".format(
                    initializer.__class__.__name__, None, model_name
                )
            )

    if mode == "evaluate":
        if os.path.exists(
            os.path.join(
                "models", model_name, "pytest_data", "pytest_file", "setup.pickle"
            )
        ):
            os.remove(
                os.path.join(
                    "models", model_name, "pytest_data", "pytest_file", "setup.pickle"
                )
            )
        if os.path.exists(
            os.path.join("models", model_name, "pytest_data", "pytest_file")
        ):
            os.rmdir(os.path.join("models", model_name, "pytest_data", "pytest_file"))
        if os.path.exists(os.path.join("models", model_name, "pytest_data")):
            os.rmdir(os.path.join("models", model_name, "pytest_data"))


@pytest.mark.parametrize("mode", modes)
@pytest.mark.parametrize("model_name", models)
def test_initializer_get_evaluater(mode, model_name):
    args = setup_user_args(mode, model_name)
    initializer = Initializer(
        architecture_settings, inference_settings, simulation_settings, **args
    )
    if mode == "train_offline":
        initializer.save_setup()
    evaluater = initializer.get_evaluater()

    assert isinstance(evaluater, Evaluater)

    if mode == "evaluate":
        if os.path.exists(
            os.path.join(
                "models", model_name, "pytest_data", "pytest_file", "setup.pickle"
            )
        ):
            os.remove(
                os.path.join(
                    "models", model_name, "pytest_data", "pytest_file", "setup.pickle"
                )
            )
        if os.path.exists(
            os.path.join("models", model_name, "pytest_data", "pytest_file")
        ):
            os.rmdir(os.path.join("models", model_name, "pytest_data", "pytest_file"))
        if os.path.exists(os.path.join("models", model_name, "pytest_data")):
            os.rmdir(os.path.join("models", model_name, "pytest_data"))


@pytest.mark.parametrize("mode", modes)
@pytest.mark.parametrize("model_name", models)
def test_initializer_save_losses(mode, model_name, capsys):
    args = setup_user_args(mode, model_name)
    initializer = Initializer(
        architecture_settings, inference_settings, simulation_settings, **args
    )
    num_iterations = random.randint(1, 10)
    dummy_losses = pd.DataFrame(
        np.random.uniform(-100, 100, size=(num_iterations, 1)), columns=["Default.Loss"]
    )

    if mode == "train_online":
        initializer.save_losses(dummy_losses)
        assert os.path.join(
            "results", model_name, "online", "pytest_file", "losses.pickle"
        )
        if os.path.exists(
            os.path.join(
                "results", model_name, "online", "pytest_file", "losses.pickle"
            )
        ):
            os.remove(
                os.path.join(
                    "results", model_name, "online", "pytest_file", "losses.pickle"
                )
            )
        if os.path.join(os.path.join("results", model_name, "online", "pytest_file")):
            os.rmdir(os.path.join("results", model_name, "online", "pytest_file"))
    elif mode == "train_offline":
        initializer.save_setup()
        initializer.save_losses(dummy_losses)
        assert os.path.exists(
            os.path.join(
                "results", model_name, "pytest_data", "pytest_file", "losses.pickle"
            )
        )
    else:
        with pytest.raises(ValueError):
            initializer.save_losses(dummy_losses)
            out, err = capsys.readouterr()
            assert out == ""
            assert (
                err
                == "{} - save_losses: main.py was executed in {} mode, but needs to be in train_online or train_offline mode".format(
                    initializer.__class__.__name__, mode
                )
            )

    if mode == "evaluate":
        if os.path.exists(
            os.path.join(
                "models", model_name, "pytest_data", "pytest_file", "setup.pickle"
            )
        ):
            os.remove(
                os.path.join(
                    "models", model_name, "pytest_data", "pytest_file", "setup.pickle"
                )
            )
        if os.path.exists(
            os.path.join("models", model_name, "pytest_data", "pytest_file")
        ):
            os.rmdir(os.path.join("models", model_name, "pytest_data", "pytest_file"))
        if os.path.exists(os.path.join("models", model_name, "pytest_data")):
            os.rmdir(os.path.join("models", model_name, "pytest_data"))


@pytest.mark.parametrize("mode", modes)
@pytest.mark.parametrize("model_name", models)
def test_initializer_save_load_setup(mode, model_name, capsys):
    args = setup_user_args(mode, model_name)
    initializer = Initializer(
        architecture_settings, inference_settings, simulation_settings, **args
    )

    if mode == "train_online":
        initializer.save_setup()
        assert os.path.exists(
            os.path.join("models", model_name, "online", "pytest_file", "setup.pickle")
        )
        initializer.summary_net_name = None
        initializer.architecture = None
        initializer.inference = None
        initializer.simulation = None
        (
            initializer.summary_net_name,
            initializer.architecture,
            initializer.inference,
            initializer.simulation,
        ) = initializer.load_setup()
        assert isinstance(initializer.summary_net_name, str)
        assert isinstance(initializer.architecture, dict)
        assert isinstance(initializer.inference, dict)
        assert isinstance(initializer.simulation, dict)
        assert initializer.summary_net_name == "FC"
        assert initializer.inference == inference_settings[model_name]
        assert initializer.simulation == simulation_settings[model_name]
        if os.path.exists(
            os.path.join("models", model_name, "online", "pytest_file", "setup.pickle")
        ):
            os.remove(
                os.path.join(
                    "models", model_name, "online", "pytest_file", "setup.pickle"
                )
            )
        if os.path.exists(os.path.join("models", model_name, "online", "pytest_file")):
            os.rmdir(os.path.join("models", model_name, "online", "pytest_file"))
    elif mode == "train_offline":
        initializer.save_setup()
        assert os.path.exists(
            os.path.join(
                "models", model_name, "pytest_data", "pytest_file", "setup.pickle"
            )
        )
        initializer.summary_net_name = None
        initializer.architecture = None
        initializer.inference = None
        initializer.simulation = None
        (
            initializer.summary_net_name,
            initializer.architecture,
            initializer.inference,
            initializer.simulation,
        ) = initializer.load_setup()
        assert isinstance(initializer.summary_net_name, str)
        assert isinstance(initializer.architecture, dict)
        assert isinstance(initializer.inference, dict)
        assert isinstance(initializer.simulation, dict)
        assert initializer.summary_net_name == "FC"
        assert initializer.inference == inference_settings[model_name]
        assert initializer.simulation == simulation_settings[model_name]
    else:
        with pytest.raises(ValueError):
            initializer.save_setup()
            out, err = capsys.readouterr()
            assert out == ""
            assert (
                err
                == "{} - save_setup: main.py was executed in {} mode, but needs to be in train_online or train_offline mode".format(
                    initializer.__class__.__name__, mode
                )
            )

    if mode == "evaluate":
        if os.path.exists(
            os.path.join(
                "models", model_name, "pytest_data", "pytest_file", "setup.pickle"
            )
        ):
            os.remove(
                os.path.join(
                    "models", model_name, "pytest_data", "pytest_file", "setup.pickle"
                )
            )
        if os.path.exists(
            os.path.join("models", model_name, "pytest_data", "pytest_file")
        ):
            os.rmdir(os.path.join("models", model_name, "pytest_data", "pytest_file"))
        if os.path.exists(os.path.join("models", model_name, "pytest_data")):
            os.rmdir(os.path.join("models", model_name, "pytest_data"))


@pytest.mark.parametrize("mode", modes)
@pytest.mark.parametrize("model_name", models)
def test_initializer_generate_load_hdf5(mode, model_name, capsys):
    args = setup_user_args(mode, model_name)
    initializer = Initializer(
        architecture_settings, inference_settings, simulation_settings, **args
    )
    if mode == "train_offline":
        initializer.save_setup()

    if mode == "generate_data":
        initializer.generate_hdf5_data()
        data_dict = initializer.load_hdf5_data()
        assert os.path.exists(
            os.path.join("data", model_name, "pytest_data", "data.h5")
        )
        assert isinstance(data_dict, dict)
        if os.path.exists(os.path.join("data", model_name, "pytest_data", "data.h5")):
            os.remove(os.path.join("data", model_name, "pytest_data", "data.h5"))
        if os.path.exists(os.path.join("data", model_name, "pytest_data")):
            os.rmdir(os.path.join("data", model_name, "pytest_data"))
    else:
        with pytest.raises(ValueError):
            initializer.generate_hdf5_data()
            out, err = capsys.readouterr()

            assert out == ""
            assert (
                err
                == "{} - generate_hdf5_data: main.py was executed in {} mode, but needs to be in generate_data mode".format(
                    initializer.__class__.__name__, mode
                )
            )

    if mode == "evaluate":
        if os.path.exists(
            os.path.join(
                "models", model_name, "pytest_data", "pytest_file", "setup.pickle"
            )
        ):
            os.remove(
                os.path.join(
                    "models", model_name, "pytest_data", "pytest_file", "setup.pickle"
                )
            )
        if os.path.exists(
            os.path.join("models", model_name, "pytest_data", "pytest_file")
        ):
            os.rmdir(os.path.join("models", model_name, "pytest_data", "pytest_file"))
        if os.path.exists(os.path.join("models", model_name, "pytest_data")):
            os.rmdir(os.path.join("models", model_name, "pytest_data"))
