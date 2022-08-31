import os
import shutil

import pytest

from ML_for_Battery_Design.src.helpers.constants import inference_settings
from ML_for_Battery_Design.src.helpers.wrappers import train_online
from tests.helpers import setup_user_args

models = ["linear_ode_system"]


def check_file_exist(eval_key, model_name, expected_path):
    if inference_settings[model_name]["evaluation"][eval_key]:
        assert os.path.exists(expected_path)
        if os.path.exists(expected_path):
            os.remove(expected_path)


@pytest.mark.parametrize("model_name", models)
def test_wrappers_train_online(model_name):
    args = setup_user_args("train_online", model_name, save_model=True)
    train_online(**args)

    assert os.path.exists(
        os.path.join("model", model_name, "online", "pytest_file", "checkpoint")
    )
    if os.path.exists(os.path.join("model", model_name, "online", "pytest_file")):
        shutil.rmtree(os.path.join("model", model_name, "online", "pytest_file"))

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
