import os
import random
import string

import pytest
import tensorflow as tf
from bayesflow.amortized_inference import AmortizedPosterior
from bayesflow.networks import InvertibleNetwork
from bayesflow.trainers import Trainer

from ML_for_Battery_Design.src.helpers.constants import sim_model_collection
from ML_for_Battery_Design.src.helpers.evaluater import Evaluater
from ML_for_Battery_Design.src.helpers.filemanager import FileManager
from ML_for_Battery_Design.src.helpers.initializer import Initializer
from ML_for_Battery_Design.src.helpers.processing import Processing
from ML_for_Battery_Design.src.helpers.summary import FC_Network
from ML_for_Battery_Design.src.simulation.simulation_model import SimulationModel
from tests.helpers import setup_user_args

models = ["linear_ode_system"]

modes = ["train_online", "train_offline", "generate_data", "analyze_sim", "evaluate"]


@pytest.mark.parametrize("model_name", models)
def test_initializer_init_train_online(model_name):
    args = setup_user_args("train_online", model_name)

    initializer = Initializer(**args)

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
    assert initializer.sim_model_name == model_name
    assert initializer.summary_net_name == "FC"
    assert initializer.data_name == "online"
    assert initializer.filename == "pytest_file"
    assert not initializer.save_model
    assert initializer.test_mode
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
    assert isinstance(initializer.test_mode, bool)
    assert isinstance(initializer.sim_model, SimulationModel)
    assert isinstance(initializer.sim_model, sim_model_collection[model_name])
    assert isinstance(initializer.mode, str)
    assert isinstance(initializer.file_manager, FileManager)
    assert initializer.sim_model_name == model_name
    assert initializer.summary_net_name == "FC"
    assert initializer.data_name == "pytest_data"
    assert initializer.filename == "pytest_file"
    assert not initializer.save_model
    assert initializer.test_mode
    assert initializer.mode == "train_offline"
    assert isinstance(initializer.trainer, Trainer)


@pytest.mark.parametrize("model_name", models)
def test_initializer_init_generate_data(model_name):
    args = setup_user_args("generate_data", model_name)

    initializer = Initializer(**args)

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

    initializer = Initializer(**args)

    assert isinstance(initializer.sim_model_name, str)
    assert isinstance(initializer.filename, str)
    assert isinstance(initializer.save_model, bool)
    assert isinstance(initializer.test_mode, bool)
    assert isinstance(initializer.sim_model, SimulationModel)
    assert isinstance(initializer.sim_model, sim_model_collection[model_name])
    assert isinstance(initializer.mode, str)
    assert isinstance(initializer.file_manager, FileManager)
    assert initializer.sim_model_name == model_name
    assert initializer.summary_net_name is None
    assert initializer.data_name is None
    assert initializer.filename == "pytest_file"
    assert not initializer.save_model
    assert initializer.test_mode
    assert initializer.mode == "analyze_sim"


@pytest.mark.parametrize("model_name", models)
def test_initializer_init_evaluate(model_name):
    args = setup_user_args("evaluate", model_name)

    initializer = Initializer(**args)

    assert isinstance(initializer.sim_model_name, str)
    assert isinstance(initializer.data_name, str)
    assert isinstance(initializer.filename, str)
    assert isinstance(initializer.save_model, bool)
    assert isinstance(initializer.test_mode, bool)
    assert isinstance(initializer.sim_model, SimulationModel)
    assert isinstance(initializer.sim_model, sim_model_collection[model_name])
    assert isinstance(initializer.mode, str)
    assert isinstance(initializer.file_manager, FileManager)
    assert initializer.sim_model_name == model_name
    assert initializer.summary_net_name is None
    assert initializer.data_name == "pytest_data"
    assert initializer.filename == "pytest_file"
    assert not initializer.save_model
    assert initializer.test_mode
    assert initializer.mode == "evaluate"


@pytest.mark.parametrize("mode", modes)
@pytest.mark.parametrize("model_name", models)
def test_initializer_get_sim_model(mode, model_name):
    args = setup_user_args(mode, model_name)
    initializer = Initializer(**args)
    sim_model = initializer.get_sim_model()

    assert isinstance(sim_model, SimulationModel)
    assert isinstance(sim_model, sim_model_collection[model_name])


@pytest.mark.parametrize("mode", modes)
def test_initializer_get_sim_model_sim_setting_error(mode, capsys):
    dummy_model_name = "".join(
        random.choices(string.ascii_letters, k=random.randrange(10))
    )
    args = setup_user_args(mode, dummy_model_name)

    with pytest.raises(ValueError):
        initializer = Initializer(**args)
        out, err = capsys.readouterr()
        assert out == ""
        assert err == "{} - get_sim_model: {} not found in simulation settings".format(
            initializer.__class__.__name__, dummy_model_name
        )


@pytest.mark.parametrize("mode", modes)
def test_initializer_get_sim_model_sim_model_error(mode, capsys):
    args = setup_user_args(mode, "pytest")

    with pytest.raises(ValueError):
        initializer = Initializer(**args)
        out, err = capsys.readouterr()
        assert out == ""
        assert err == "{} - get_sim_model: {} is not a valid simulation model".format(
            initializer.__class__.__name__, "pytest"
        )


@pytest.mark.parametrize("mode", modes)
@pytest.mark.parametrize("model_name", models)
def test_initializer_get_summary_net(mode, model_name, capsys):
    args = setup_user_args(mode, model_name)
    initializer = Initializer(**args)

    if mode in ["train_online", "train_offline"]:
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


@pytest.mark.parametrize("mode", modes)
def test_initializer_get_summary_net_no_architecture(mode, capsys):
    dummy_model_name = "".join(
        random.choices(string.ascii_letters, k=random.randrange(10))
    )
    args = setup_user_args(mode, random.choice(models))
    initializer = Initializer(**args)
    initializer.sim_model_name = dummy_model_name
    if mode in ["train_online", "train_offline"]:
        expected_model_name = dummy_model_name
    else:
        expected_model_name = None

    with pytest.raises(ValueError):
        initializer.get_summary_net()
        out, err = capsys.readouterr()

        assert out == ""
        assert (
            err
            == "{} - get_summary_net: simulation model {} has no architecture settings".format(
                initializer.__class__.__name__, expected_model_name
            )
        )


@pytest.mark.parametrize("mode", modes)
@pytest.mark.parametrize("model_name", models)
def test_initializer_get_summary_net_invalid_sum_net(mode, model_name, capsys):
    args = setup_user_args(mode, model_name)
    initializer = Initializer(**args)
    initializer.sim_model_name = "pytest"

    if mode in ["train_online", "train_offline"]:
        initializer.summary_net_name = "pytest"
        expected_summary_name = "pytest"
    else:
        expected_summary_name = None

    with pytest.raises(ValueError):
        initializer.get_summary_net()
        out, err = capsys.readouterr()

        assert out == ""
        assert (
            err
            == "{} - get_summary_net: {} is not a valid summary network architecture".format(
                initializer.__class__.__name__, expected_summary_name
            )
        )


@pytest.mark.parametrize("mode", modes)
@pytest.mark.parametrize("model_name", models)
def test_initializer_get_inference_net(mode, model_name):
    args = setup_user_args(mode, model_name)
    initializer = Initializer(**args)
    inference_net = initializer.get_inference_net()

    assert isinstance(inference_net, InvertibleNetwork)


@pytest.mark.parametrize("mode", modes)
def test_initializer_get_inference_net_invalid_sim_model(mode, capsys):
    dummy_model_name = "".join(
        random.choices(string.ascii_letters, k=random.randrange(10))
    )
    args = setup_user_args(mode, random.choice(models))
    initializer = Initializer(**args)
    initializer.sim_model_name = dummy_model_name

    with pytest.raises(ValueError):
        initializer.get_inference_net()
        out, err = capsys.readouterr()

        assert out == ""
        assert (
            err
            == "{} - get_inference_net: {} is not a valid simulation model".format(
                initializer.__class__.__name__, dummy_model_name
            )
        )


@pytest.mark.parametrize("mode", modes)
def test_initializer_get_inference_net_no_INN_architecture(mode, capsys):
    args = setup_user_args(mode, random.choice(models))
    initializer = Initializer(**args)
    initializer.sim_model_name = "pytest"

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


@pytest.mark.parametrize("mode", modes)
@pytest.mark.parametrize("model_name", models)
def test_initializer_get_amortizer(mode, model_name, capsys):
    args = setup_user_args(mode, model_name)
    initializer = Initializer(**args)

    if mode in ["train_online", "train_offline"]:
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


@pytest.mark.parametrize("mode", modes)
@pytest.mark.parametrize("model_name", models)
def test_initializer_get_configurator(mode, model_name):
    args = setup_user_args(mode, model_name)
    initializer = Initializer(**args)
    configurator = initializer.get_configurator()

    assert isinstance(configurator, Processing)


@pytest.mark.parametrize("mode", modes)
def test_initializer_get_configurator_invalid_sim_model(mode, capsys):
    dummy_model_name = "".join(
        random.choices(string.ascii_letters, k=random.randrange(10))
    )
    args = setup_user_args(mode, random.choice(models))
    initializer = Initializer(**args)
    initializer.sim_model_name = dummy_model_name

    with pytest.raises(ValueError):
        initializer.get_configurator()
        out, err = capsys.readouterr()

        assert out == ""
        assert (
            err
            == "{} - get_configurator: {} is not a valid simulation model".format(
                initializer.__class__.__name__, dummy_model_name
            )
        )


@pytest.mark.parametrize("mode", modes)
def test_initializer_get_configurator_no_inference_settings(mode, capsys):
    args = setup_user_args(mode, random.choice(models))
    initializer = Initializer(**args)
    initializer.sim_model_name = "pytest"

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


@pytest.mark.parametrize("mode", modes)
@pytest.mark.parametrize("model_name", models)
def test_initilizer_get_trainer(mode, model_name, capsys):
    args = setup_user_args(mode, model_name)
    initializer = Initializer(**args)

    if mode in ["train_online", "train_offline"]:
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


@pytest.mark.parametrize("mode", modes)
@pytest.mark.parametrize("model_name", models)
def test_initializer_get_evaluater(mode, model_name, capsys):
    args = setup_user_args(mode, model_name)
    initializer = Initializer(**args)

    if mode in ["train_online", "train_offline"]:
        evaluater = initializer.get_evaluater()
        assert isinstance(evaluater, Evaluater)
    else:
        with pytest.raises(ValueError):
            initializer.get_evaluater()
            out, err = capsys.readouterr()
            assert out == ""
            assert (
                err
                == "{} - get_summary_net: {} not found in architecture settings for {} simulation model".format(
                    initializer.__class__.__name__, None, model_name
                )
            )


@pytest.mark.parametrize("mode", modes)
@pytest.mark.parametrize("model_name", models)
def test_initializer_generate_hdf5(mode, model_name, capsys):
    args = setup_user_args(mode, model_name)
    initializer = Initializer(**args)

    if mode == "generate_data":
        initializer.generate_hdf5_data()
        assert os.path.exists(
            os.path.join("data", model_name, "pytest_data", "data.h5")
        )
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
