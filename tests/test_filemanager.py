import os
import random
import string

import pytest

from ML_for_Battery_Design.src.helpers.filemanager import FileManager

valid_modes = [
    "train_online",
    "train_offline",
    "generate_data",
    "analyze_sim",
    "evaluate",
]

valid_sim = ["linear_ode_system"]

valid_summary = ["FC", "LSTM", "CNN"]

random_input = [
    "",
    "".join(random.choices(string.ascii_letters, k=random.randrange(10))),
]


def test_filemanager_init():
    mode = random.choice(random_input)
    sim_model_name = random.choice(random_input)
    summary_net_name = random.choice(random_input)
    data_name = random.choice(random_input)
    filename = random.choice(random_input)

    kwargs = {
        "<sim_model>": sim_model_name,
        "<summary_net>": summary_net_name,
        "<data_name>": data_name,
        "<filename>": filename,
    }

    test_object = FileManager(mode, **kwargs)

    assert isinstance(test_object.mode, str)
    assert isinstance(test_object.sim_model_name, str)
    assert isinstance(test_object.summary_net_name, str)
    assert isinstance(test_object.data_name, str)
    assert isinstance(test_object.filename, str)
    assert test_object.mode == mode
    assert test_object.sim_model_name == sim_model_name
    assert test_object.summary_net_name == summary_net_name
    assert test_object.data_name == data_name
    assert test_object.filename == filename


@pytest.mark.parametrize("file_type", ["data", "model", "result"])
def test_filemanager_call(file_type):
    mode = random.choice(random_input)
    sim_model_name = random.choice(random_input)
    summary_net_name = random.choice(random_input)
    data_name = random.choice(random_input)
    filename = random.choice(random_input)

    kwargs = {
        "<sim_model>": sim_model_name,
        "<summary_net>": summary_net_name,
        "<data_name>": data_name,
        "<filename>": filename,
    }

    test_object = FileManager(mode, **kwargs)

    path = test_object(file_type)

    assert isinstance(path, str)
    if file_type == "data":
        assert path == os.path.join("data", sim_model_name, data_name)
    elif file_type == "model":
        assert path == os.path.join("models", sim_model_name, data_name, filename)
    elif file_type == "result":
        assert path == os.path.join("results", sim_model_name, data_name, filename)
    else:
        assert False


@pytest.mark.parametrize("file_type", random_input)
def test_filemanager_call_not_valid_file_type(file_type, capsys):
    mode = random.choice(random_input)
    sim_model_name = random.choice(random_input)
    summary_net_name = random.choice(random_input)
    data_name = random.choice(random_input)
    filename = random.choice(random_input)

    kwargs = {
        "<sim_model>": sim_model_name,
        "<summary_net>": summary_net_name,
        "<data_name>": data_name,
        "<filename>": filename,
    }

    test_object = FileManager(mode, **kwargs)

    with pytest.raises(ValueError):
        test_object(file_type)
        out, err = capsys.readouterr()
        assert out == ""
        assert (
            err
            == "{} - call: {} is not a valid file_type. Valid options are: data, model, results".format(
                test_object.__class__.__name__, file_type
            )
        )
