import os
import shutil

import pytest

from ML_for_Battery_Design.src.helpers.wrappers import (
    analyze_sim,
    evaluate,
    generate_data,
    train_offline,
    train_online,
)
from tests.constants import models
from tests.helpers import check_file_exist, setup_user_args


@pytest.mark.parametrize("model_name", models)
def test_wrappers_train_online(model_name):
    args = setup_user_args("train_online", model_name, save_model=True)
    args["--skip_wrappers"] = False
    train_online(**args)

    assert os.path.exists(
        os.path.join("models", model_name, "online", "pytest_file", "checkpoint")
    )

    assert os.path.exists(
        os.path.join("results", model_name, "online", "pytest_file", "losses.pickle")
    )
    if os.path.exists(
        os.path.join("results", model_name, "online", "pytest_file", "losses.pickle")
    ):
        os.remove(
            os.path.join(
                "results", model_name, "online", "pytest_file", "losses.pickle"
            )
        )

    assert os.path.exists(
        os.path.join("models", model_name, "online", "pytest_file", "setup.pickle")
    )
    if os.path.exists(
        os.path.join("models", model_name, "online", "pytest_file", "setup.pickle")
    ):
        os.remove(
            os.path.join("models", model_name, "online", "pytest_file", "setup.pickle")
        )

    check_file_exist(
        "plot_prior",
        model_name,
        os.path.join("results", model_name, "online", "pytest_file", "prior_2d.png"),
    )

    check_file_exist(
        "plot_sim_data",
        model_name,
        os.path.join("results", model_name, "online", "pytest_file", "sim_data.png"),
    )

    check_file_exist(
        "plot_loss",
        model_name,
        os.path.join("results", model_name, "online", "pytest_file", "loss.png"),
    )

    check_file_exist(
        "plot_latent",
        model_name,
        os.path.join("results", model_name, "online", "pytest_file", "latent_2d.png"),
    )

    check_file_exist(
        "plot_sbc_histogram",
        model_name,
        os.path.join("results", model_name, "online", "pytest_file", "sbc_hist.png"),
    )

    check_file_exist(
        "plot_sbc_ecdf",
        model_name,
        os.path.join("results", model_name, "online", "pytest_file", "sbc_ecdf.png"),
    )

    check_file_exist(
        "plot_true_vs_estimated",
        model_name,
        os.path.join(
            "results", model_name, "online", "pytest_file", "true_vs_estimated.png"
        ),
    )

    check_file_exist(
        "plot_posterior",
        model_name,
        os.path.join("results", model_name, "online", "pytest_file", "posterior.png"),
    )

    check_file_exist(
        "plot_post_with_prior",
        model_name,
        os.path.join(
            "results", model_name, "online", "pytest_file", "compare_prior_post.png"
        ),
    )

    check_file_exist(
        "plot_resimulation",
        model_name,
        os.path.join(
            "results", model_name, "online", "pytest_file", "resimulation.png"
        ),
    )
    if os.path.exists(os.path.join("results", model_name, "online", "pytest_file")):
        os.rmdir(os.path.join("results", model_name, "online", "pytest_file"))
    if os.path.exists(os.path.join("models", model_name, "online", "pytest_file")):
        shutil.rmtree(os.path.join("models", model_name, "online", "pytest_file"))


@pytest.mark.parametrize("model_name", models)
def test_wrappers_generate_data_train_offline_evaluate(model_name):
    args = setup_user_args("generate_data", model_name)
    args["--skip_wrappers"] = False
    generate_data(**args)
    args = setup_user_args("train_offline", model_name, save_model=True)
    args["--skip_wrappers"] = False
    train_offline(**args)

    assert os.path.exists(os.path.join("data", model_name, "pytest_data", "data.h5"))
    if os.path.join("data", model_name, "pytest_data", "data.h5"):
        os.remove(os.path.join("data", model_name, "pytest_data", "data.h5"))
    if os.path.join("data", model_name, "pytest_data"):
        os.rmdir(os.path.join("data", model_name, "pytest_data"))

    assert os.path.exists(
        os.path.join("models", model_name, "pytest_data", "pytest_file", "checkpoint")
    )

    assert os.path.exists(
        os.path.join(
            "results", model_name, "pytest_data", "pytest_file", "losses.pickle"
        )
    )

    assert os.path.exists(
        os.path.join("models", model_name, "pytest_data", "pytest_file", "setup.pickle")
    )

    check_file_exist(
        "plot_prior",
        model_name,
        os.path.join(
            "results", model_name, "pytest_data", "pytest_file", "prior_2d.png"
        ),
    )

    check_file_exist(
        "plot_sim_data",
        model_name,
        os.path.join(
            "results", model_name, "pytest_data", "pytest_file", "sim_data.png"
        ),
    )

    check_file_exist(
        "plot_loss",
        model_name,
        os.path.join("results", model_name, "pytest_data", "pytest_file", "loss.png"),
    )

    check_file_exist(
        "plot_latent",
        model_name,
        os.path.join(
            "results", model_name, "pytest_data", "pytest_file", "latent_2d.png"
        ),
    )

    check_file_exist(
        "plot_sbc_histogram",
        model_name,
        os.path.join(
            "results", model_name, "pytest_data", "pytest_file", "sbc_hist.png"
        ),
    )

    check_file_exist(
        "plot_sbc_ecdf",
        model_name,
        os.path.join(
            "results", model_name, "pytest_data", "pytest_file", "sbc_ecdf.png"
        ),
    )

    check_file_exist(
        "plot_true_vs_estimated",
        model_name,
        os.path.join(
            "results", model_name, "pytest_data", "pytest_file", "true_vs_estimated.png"
        ),
    )

    check_file_exist(
        "plot_posterior",
        model_name,
        os.path.join(
            "results", model_name, "pytest_data", "pytest_file", "posterior.png"
        ),
    )

    check_file_exist(
        "plot_post_with_prior",
        model_name,
        os.path.join(
            "results",
            model_name,
            "pytest_data",
            "pytest_file",
            "compare_prior_post.png",
        ),
    )

    check_file_exist(
        "plot_resimulation",
        model_name,
        os.path.join(
            "results", model_name, "pytest_data", "pytest_file", "resimulation.png"
        ),
    )

    args = setup_user_args("evaluate", model_name, save_model=True)
    args["--skip_wrappers"] = False
    evaluate(**args)

    check_file_exist(
        "plot_loss",
        model_name,
        os.path.join("results", model_name, "pytest_data", "pytest_file", "loss.png"),
    )

    check_file_exist(
        "plot_latent",
        model_name,
        os.path.join(
            "results", model_name, "pytest_data", "pytest_file", "latent_2d.png"
        ),
    )

    check_file_exist(
        "plot_sbc_histogram",
        model_name,
        os.path.join(
            "results", model_name, "pytest_data", "pytest_file", "sbc_hist.png"
        ),
    )

    check_file_exist(
        "plot_sbc_ecdf",
        model_name,
        os.path.join(
            "results", model_name, "pytest_data", "pytest_file", "sbc_ecdf.png"
        ),
    )

    check_file_exist(
        "plot_true_vs_estimated",
        model_name,
        os.path.join(
            "results", model_name, "pytest_data", "pytest_file", "true_vs_estimated.png"
        ),
    )

    check_file_exist(
        "plot_posterior",
        model_name,
        os.path.join(
            "results", model_name, "pytest_data", "pytest_file", "posterior.png"
        ),
    )

    check_file_exist(
        "plot_post_with_prior",
        model_name,
        os.path.join(
            "results",
            model_name,
            "pytest_data",
            "pytest_file",
            "compare_prior_post.png",
        ),
    )

    check_file_exist(
        "plot_resimulation",
        model_name,
        os.path.join(
            "results", model_name, "pytest_data", "pytest_file", "resimulation.png"
        ),
    )

    if os.path.exists(
        os.path.join(
            "results", model_name, "pytest_data", "pytest_file", "losses.pickle"
        )
    ):
        os.remove(
            os.path.join(
                "results", model_name, "pytest_data", "pytest_file", "losses.pickle"
            )
        )

    if os.path.exists(
        os.path.join("models", model_name, "pytest_data", "pytest_file", "setup.pickle")
    ):
        os.remove(
            os.path.join(
                "models", model_name, "pytest_data", "pytest_file", "setup.pickle"
            )
        )

    if os.path.exists(
        os.path.join("results", model_name, "pytest_data", "pytest_file")
    ):
        os.rmdir(os.path.join("results", model_name, "pytest_data", "pytest_file"))

    if os.path.exists(os.path.join("results", model_name, "pytest_data")):
        os.rmdir(os.path.join("results", model_name, "pytest_data"))

    if os.path.exists(os.path.join("models", model_name, "pytest_data", "pytest_file")):
        shutil.rmtree(os.path.join("models", model_name, "pytest_data", "pytest_file"))


@pytest.mark.parametrize("model_name", models)
def test_wrappers_analyze_sim(model_name):
    args = setup_user_args("analyze_sim", model_name)
    args["--skip_wrappers"] = False
    analyze_sim(**args)

    check_file_exist(
        "plot_prior",
        model_name,
        os.path.join("results", model_name, "prior_2d.png"),
    )

    check_file_exist(
        "plot_sim_data",
        model_name,
        os.path.join("results", model_name, "sim_data.png"),
    )
