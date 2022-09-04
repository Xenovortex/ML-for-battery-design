import random
import string

import pytest
from docopt import DocoptExit
from tabulate import tabulate

import ML_for_Battery_Design.src.main as main

random_input = [
    "",
    "".join(random.choices(string.ascii_letters, k=random.randrange(10))),
]


@pytest.mark.parametrize("filename", random_input + [None])
def test_main_train_online(filename, capsys):
    sim = random.choice(random_input)
    summary = random.choice(random_input)
    if filename is not None:
        args = main.main(
            ["train_online", sim, summary, filename, "--test_mode", "--skip_wrappers"]
        )
    else:
        args = main.main(
            ["train_online", sim, summary, "--test_mode", "--skip_wrappers"]
        )
    out, err = capsys.readouterr()
    assert (
        out
        == "Interface user input:\n"
        + tabulate(list(args.items()), missingval="None")
        + "\n"
    )
    assert err == ""
    assert bool(args["train_online"])
    assert not bool(args["train_offline"])
    assert not bool(args["generate_data"])
    assert not bool(args["analyze_sim"])
    assert not bool(args["evaluate"])
    assert args["<sim_model>"] == sim
    assert args["<summary_net>"] == summary
    assert args["<filename>"] == filename
    assert args["<data_name>"] is None
    assert not args["--save_model"]
    assert args["--test_mode"]


@pytest.mark.parametrize("filename", random_input + [None])
@pytest.mark.parametrize("save_model", ["-s", "--save_model"])
def test_main_train_online_save_model(filename, save_model, capsys):
    sim = random.choice(random_input)
    summary = random.choice(random_input)
    if filename is None:
        with pytest.raises(DocoptExit):
            main.main(
                [
                    "train_online",
                    sim,
                    summary,
                    save_model,
                    "--test_mode",
                    "--skip_wrappers",
                ]
            )
    else:
        args = main.main(
            [
                "train_online",
                sim,
                summary,
                filename,
                save_model,
                "--test_mode",
                "--skip_wrappers",
            ]
        )
        out, err = capsys.readouterr()
        assert out == (
            "Interface user input:\n"
            + tabulate(list(args.items()), missingval="None")
            + "\n"
        )
        assert err == ""
        assert bool(args["train_online"])
        assert not bool(args["train_offline"])
        assert not bool(args["generate_data"])
        assert not bool(args["analyze_sim"])
        assert not bool(args["evaluate"])
        assert args["<sim_model>"] == sim
        assert args["<summary_net>"] == summary
        assert args["<filename>"] == filename
        assert args["<data_name>"] is None
        assert args["--save_model"]
        assert args["--test_mode"]


@pytest.mark.parametrize("filename", random_input + [None])
def test_main_train_offline(filename, capsys):
    sim = random.choice(random_input)
    data = random.choice(random_input)
    summary = random.choice(random_input)
    if filename is not None:
        args = main.main(
            [
                "train_offline",
                sim,
                data,
                summary,
                filename,
                "--test_mode",
                "--skip_wrappers",
            ]
        )
    else:
        args = main.main(
            ["train_offline", sim, data, summary, "--test_mode", "--skip_wrappers"]
        )
    out, err = capsys.readouterr()
    assert (
        out
        == "Interface user input:\n"
        + tabulate(list(args.items()), missingval="None")
        + "\n"
    )
    assert err == ""
    assert bool(args["train_offline"])
    assert not bool(args["train_online"])
    assert not bool(args["generate_data"])
    assert not bool(args["analyze_sim"])
    assert not bool(args["evaluate"])
    assert args["<sim_model>"] == sim
    assert args["<data_name>"] == data
    assert args["<summary_net>"] == summary
    assert args["<filename>"] == filename
    assert not args["--save_model"]


@pytest.mark.parametrize("filename", random_input + [None])
@pytest.mark.parametrize("save_model", ["-s", "--save_model"])
def test_main_train_offline_save_model(filename, save_model, capsys):
    sim = random.choice(random_input)
    data = random.choice(random_input)
    summary = random.choice(random_input)
    if filename is None:
        with pytest.raises(DocoptExit):
            main.main(
                [
                    "train_offline",
                    sim,
                    summary,
                    save_model,
                    "--test_mode",
                    "--skip_wrappers",
                ]
            )
    else:
        args = main.main(
            [
                "train_offline",
                sim,
                data,
                summary,
                filename,
                save_model,
                "--test_mode",
                "--skip_wrappers",
            ]
        )
        out, err = capsys.readouterr()
        assert (
            out
            == "Interface user input:\n"
            + tabulate(list(args.items()), missingval="None")
            + "\n"
        )
        assert err == ""
        assert bool(args["train_offline"])
        assert not bool(args["train_online"])
        assert not bool(args["generate_data"])
        assert not bool(args["analyze_sim"])
        assert not bool(args["evaluate"])
        assert args["<sim_model>"] == sim
        assert args["<data_name>"] == data
        assert args["<summary_net>"] == summary
        assert args["<filename>"] == filename
        assert args["--save_model"]


def test_main_generate_data(capsys):
    sim = random.choice(random_input)
    data = random.choice(random_input)
    args = main.main(["generate_data", sim, data, "--test_mode", "--skip_wrappers"])
    out, err = capsys.readouterr()
    assert (
        out
        == "Interface user input:\n"
        + tabulate(list(args.items()), missingval="None")
        + "\n"
    )
    assert err == ""
    assert bool(args["generate_data"])
    assert not bool(args["train_online"])
    assert not bool(args["train_offline"])
    assert not bool(args["analyze_sim"])
    assert not bool(args["evaluate"])
    assert args["<sim_model>"] == sim
    assert args["<data_name>"] == data
    assert args["<summary_net>"] is None
    assert args["<filename>"] is None
    assert not args["--save_model"]


@pytest.mark.parametrize("filename", random_input + [None])
def test_main_analyze_sim(filename, capsys):
    sim = random.choice(random_input)
    if filename is not None:
        args = main.main(
            [
                "analyze_sim",
                sim,
                "pytest_data",
                filename,
                "--test_mode",
                "--skip_wrappers",
            ]
        )
    else:
        args = main.main(["analyze_sim", sim, "--test_mode", "--skip_wrappers"])
    out, err = capsys.readouterr()
    assert (
        out
        == "Interface user input:\n"
        + tabulate(list(args.items()), missingval="None")
        + "\n"
    )
    assert err == ""
    assert bool(args["analyze_sim"])
    assert not bool(args["train_online"])
    assert not bool(args["train_offline"])
    assert not bool(args["generate_data"])
    assert not bool(args["evaluate"])
    assert args["<sim_model>"] == sim
    assert args["<filename>"] == filename
    if filename is None:
        assert args["<data_name>"] is None
    else:
        assert args["<data_name>"] == "pytest_data"
    assert args["<summary_net>"] is None
    assert not args["--save_model"]


@pytest.mark.parametrize("filename", random_input)
def test_main_evaluate(filename, capsys):
    sim = random.choice(random_input)
    data = random.choice(random_input)
    args = main.main(
        ["evaluate", sim, data, filename, "--test_mode", "--skip_wrappers"]
    )
    out, err = capsys.readouterr()
    assert (
        out
        == "Interface user input:\n"
        + tabulate(list(args.items()), missingval="None")
        + "\n"
    )
    assert err == ""
    assert bool(args["evaluate"])
    assert not bool(args["train_online"])
    assert not bool(args["train_offline"])
    assert not bool(args["generate_data"])
    assert not bool(args["analyze_sim"])
    assert args["<sim_model>"] == sim
    assert args["<data_name>"] == data
    assert args["<filename>"] == filename
    assert args["<summary_net>"] is None
    assert not args["--save_model"]


@pytest.mark.parametrize("random_input", random_input)
def test_main_random_input(random_input):
    with pytest.raises(DocoptExit):
        main.main([random_input] * random.randrange(10))
