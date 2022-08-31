import os
import pathlib
import time
from typing import Callable, Type, Union

import bayesflow.diagnostics as diag
import matplotlib.pyplot as plt
import pandas as pd
from bayesflow.trainers import Trainer

from ML_for_Battery_Design.src.simulation.simulation_model import SimulationModel


class Evaluater:
    """Evaluates simulation model and trained BayesFlow model

    Attributes:
        sim_model (Type[SimulationModel]): simulation model allows for prior sampling and simulation data generation
        amortizer (Type[AmortizedPosterior]): BayesFlow model consisting of summary network and cINN
        trainer (Type[Trainer]): trainer for optimizing BayesFlow model
        plot_settings (dict): settings for plotting
        eval_settings (dict): settings for evaluation
        losses (pd.DataFrame or str): recorded training loss
        test_dict (dict): contains testing data for evaluation
    """

    def __init__(
        self,
        sim_model: Type[SimulationModel],
        trainer: Type[Trainer],
        plot_settings: dict,
        evaluation_settings: dict,
    ) -> None:
        """Initializes :class:Evaluater

        Args:
            sim_model (Type[SimulationModel]): simulation model allows for prior sampling and simulation data generation
            amortizer (Type[AmortizedPosterior]): BayesFlow model consisting of summary network and cINN
            trainer (Type[Trainer]): trainer for optimizing BayesFlow model
            plot_settings (dict): settings for plotting
            evaluation_settings (dict): settings for evaluation
        """
        self.sim_model = sim_model
        self.amortizer = trainer.amortizer
        self.trainer = trainer
        self.plot_settings = plot_settings
        self.eval_settings = evaluation_settings
        self.losses: Union[Type[pd.DataFrame], str]

        self.test_dict = self.generate_test_data(
            self.eval_settings["batch_size"], self.eval_settings["n_samples"]
        )

    def evaluate_sim_model(self, parent_folder: str = None) -> None:
        """Evaluates simulation model

        Args:
            parent_folder (str, optional): path to parent folder to save generated plots. Defaults to None.
        """
        if self.eval_settings["plot_prior"]:
            print("Plotting prior...")
            self.sim_model.plot_prior2d(parent_folder)

        if self.eval_settings["plot_sim_data"]:
            print("Plotting simulation data ...")
            self.sim_model.plot_sim_data(parent_folder)

    def load_losses(self, losses: Union[Type[pd.DataFrame], str]) -> None:
        """Load recorded losses, either as DataFrame or path to pickle file

        Args:
            losses (Union[Type[pd.DataFrame], str]): DataFrame or path to pickle file containing training losses
        """
        if isinstance(losses, pd.DataFrame):
            self.losses = losses
        elif isinstance(losses, str):
            with open(losses, "rb") as file:
                self.losses = pd.read_pickle(file)
        else:
            raise TypeError(
                "{} - load_losses: argument losses is {}, but has to be string or dict".format(
                    self.__class__.__name__, type(losses)
                )
            )

    def plot_wrapper(
        self,
        plot_func: Callable,
        parent_folder: str = None,
        title_name: str = None,
        filename: str = None,
        **plot_func_kwargs
    ) -> None:
        """Wrapper for optional saving and showing of generated plots

        Args:
            plot_func (Callable): function to generate plot
            parent_folder (str, optional): path to parent folder to save generated plot. Defaults to None.
            title_name (str, optional): add title to generated plot. Defaults to None.
            filename (str, optional): filename of generated plot. Defaults to None.
        """
        plt.rcParams["font.size"] = self.plot_settings["font_size"]
        fig = plot_func(**plot_func_kwargs)
        if self.plot_settings["show_title"]:
            if title_name is not None:
                fig.suptitle("{}".format(title_name))
            else:
                raise ValueError(
                    "{} - plot_wrapper: plot_setting['show_title'] is True, but no title_name is None".format(
                        self.__class__.__name__
                    )
                )
        if parent_folder is not None:
            if filename is not None:
                pathlib.Path(parent_folder).mkdir(parents=True, exist_ok=True)
                save_path = os.path.join(parent_folder, filename)
                fig.savefig(
                    save_path, transparent=True, bbox_inches="tight", pad_inches=0
                )
            else:
                raise ValueError(
                    "{} - plot_wrapper: parent_folder {} given, but filename is None".format(
                        self.__class__.__name__, parent_folder
                    )
                )
        if self.plot_settings["show_plot"]:
            plt.show(block=True if self.plot_settings["show_time"] is None else False)
            if self.plot_settings["show_time"] is not None:
                time.sleep(self.plot_settings["show_time"])
            plt.close()

    def generate_test_data(self, batch_size: int = 300, n_samples: int = 100) -> dict:
        """Generate test data for evaluation

        Args:
            batch_size (int, optional): batch size of test data. Defaults to 300.
            n_samples (int, optional): number of samples of test data for each prior combination. Defaults to 100.

        Returns:
            test_dict (dict): contains testing data for evaluation
        """
        test_data_raw = self.sim_model.generative_model(batch_size=batch_size)
        test_data = self.trainer.configurator(test_data_raw)
        posterior_samples = self.amortizer.sample(test_data, n_samples=n_samples)
        posterior_samples_unnorm = (
            self.sim_model.prior_stds * posterior_samples + self.sim_model.prior_means
        )

        test_dict = {
            "test_data_raw": test_data_raw,
            "test_data_process": test_data,
            "posterior_samples": posterior_samples,
            "posterior_samples_unnorm": posterior_samples_unnorm,
        }

        return test_dict

    def evaluate_bayesflow_model(self, parent_folder: str = None) -> None:
        """Evaluate trained BayesFlow model

        Args:
            parent_folder (str, optional): path to parent folder to save generated plots. Defaults to None.
        """
        if self.eval_settings["plot_loss"]:
            print("Plotting training loss...")
            kwargs = {"history": self.losses}
            self.plot_wrapper(
                diag.plot_losses, parent_folder, "training loss", "loss.png", **kwargs
            )

        if self.eval_settings["plot_latent"]:
            print("Plotting latent space...")
            kwargs = {"inputs": self.test_dict["test_data_raw"]}
            self.plot_wrapper(
                self.trainer.diagnose_latent2d,
                parent_folder,
                "latent space",
                "latent_2d.png",
                **kwargs,
            )

        if self.eval_settings["plot_sbc_histogram"]:
            print("Plotting simulation-based calibration histogram...")
            kwargs = {"inputs": self.test_dict["test_data_raw"]}
            self.plot_wrapper(
                self.trainer.diagnose_sbc_histograms,
                parent_folder,
                "simulation-based calibration",
                "sbc_hist.png",
                **kwargs,
            )

        if self.eval_settings["plot_sbc_ecdf"]:
            print(
                "Plotting simulation-based calibration empirical cumulative distribution functions..."
            )
            kwargs = {
                "prior_samples": self.test_dict["test_data_process"]["parameters"],
                "post_samples": self.test_dict["posterior_samples"],
            }
            self.plot_wrapper(
                diag.plot_sbc_ecdf,
                parent_folder,
                "empirical cumulative distribution functions",
                "sbc_ecdf.png",
                **kwargs,
            )

        if self.eval_settings["plot_true_vs_estimated"]:
            print("Plotting true vs estimated...")
            kwargs = {
                "prior_samples": self.test_dict["test_data_process"]["parameters"],
                "post_samples": self.test_dict["posterior_samples"],
                "param_names": self.sim_model.hidden_param_names,
            }
            self.plot_wrapper(
                diag.plot_recovery,
                parent_folder,
                "true vs estimated",
                "true_vs_estimated.png",
                **kwargs,
            )

        if self.eval_settings["plot_posterior"]:
            print("Plotting posterior...")
            kwargs = {
                "posterior_draws": self.test_dict["posterior_samples_unnorm"][0],
                "param_names": self.sim_model.hidden_param_names,
            }
            self.plot_wrapper(
                diag.plot_posterior_2d,
                parent_folder,
                "posterior",
                "posterior.png",
                **kwargs,
            )

        if self.eval_settings["plot_post_with_prior"]:
            print("Plotting prior and posterior comparison...")
            kwargs = {
                "prior": self.sim_model.prior,
                "posterior_draws": self.test_dict["posterior_samples_unnorm"][0],
            }
            self.plot_wrapper(
                diag.plot_posterior_2d,
                parent_folder,
                "prior and posterior comparison",
                "compare_prior_post.png",
                **kwargs,
            )

        if self.eval_settings["plot_resimulation"]:
            print("Ploting resimulation...")
            self.sim_model.plot_resimulation(
                self.test_dict["posterior_samples_unnorm"],
                self.test_dict["test_data_raw"]["sim_data"],
                parent_folder,
            )
