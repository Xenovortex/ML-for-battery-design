import os
import pathlib
import time
from typing import Callable, Type, Union

import bayesflow.diagnostics as diag
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from bayesflow.trainers import Trainer
from matplotlib.lines import Line2D

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
        plot_settings: dict,
        evaluation_settings: dict,
        trainer: Type[Trainer],
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
        self.plot_settings = plot_settings
        self.eval_settings = evaluation_settings
        self.trainer = trainer
        if self.trainer is not None:
            self.amortizer = trainer.amortizer
        self.losses: Union[Type[pd.DataFrame], str, bool] = False

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

    def load_losses(self, losses: Union[Type[pd.DataFrame], str, bool]) -> None:
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
        **plot_func_kwargs,
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
        if self.trainer is None:
            raise ValueError(
                "{} - evaluate_bayesflow_model: no trainer provided".format(
                    self.__class__.__name__
                )
            )

        self.test_dict = self.generate_test_data(
            self.eval_settings["batch_size"], self.eval_settings["n_samples"]
        )

        if self.eval_settings["plot_loss"]:
            if isinstance(self.losses, bool):
                raise ValueError(
                    "{} - evaluate_bayesflow_model: no losses provided".format(
                        self.__class__.__name__
                    )
                )
            print("Plotting training loss...")
            self.losses.rename(
                {"Default.Loss": "Training Loss"}, axis="columns", inplace=True
            )
            kwargs = {"history": self.losses}
            self.plot_wrapper(diag.plot_losses, parent_folder, "", "loss.png", **kwargs)

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
            # kwargs = {"inputs": self.test_dict["test_data_raw"]}
            self.plot_wrapper(
                self.trainer.diagnose_sbc_histograms,
                parent_folder,
                "simulation-based calibration",
                "sbc_hist.png",
                # **kwargs,
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
                "prior_samples": self.test_dict["test_data_raw"]["prior_draws"],
                "post_samples": self.test_dict["posterior_samples_unnorm"],
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
            num_plots = 20
            for i in range(num_plots):
                kwargs = {
                    "posterior_draws": self.test_dict["posterior_samples_unnorm"][i],
                    "param_names": self.sim_model.hidden_param_names,
                    "ground_truth": np.expand_dims(
                        self.test_dict["test_data_raw"]["prior_draws"][i], axis=0
                    ),
                    "show_mean": True,
                }
                self.plot_wrapper(
                    self.plot_posterior_2d,
                    parent_folder,
                    "posterior",
                    "posterior_truth_mean_{}.png".format(i),
                    **kwargs,
                )
                kwargs = {
                    "posterior_draws": self.test_dict["posterior_samples_unnorm"][i],
                    "param_names": self.sim_model.hidden_param_names,
                    "ground_truth": np.expand_dims(
                        self.test_dict["test_data_raw"]["prior_draws"][i], axis=0
                    ),
                    "show_mean": False,
                }
                self.plot_wrapper(
                    self.plot_posterior_2d,
                    parent_folder,
                    "posterior",
                    "posterior_truth_{}.png".format(i),
                    **kwargs,
                )
                kwargs = {
                    "posterior_draws": self.test_dict["posterior_samples_unnorm"][i],
                    "param_names": self.sim_model.hidden_param_names,
                    "ground_truth": None,
                    "show_mean": False,
                }
                self.plot_wrapper(
                    self.plot_posterior_2d,
                    parent_folder,
                    "posterior",
                    "posterior_{}.png".format(i),
                    **kwargs,
                )

        if self.eval_settings["plot_post_with_prior"]:
            print("Plotting prior and posterior comparison...")
            num_plots = 20
            for i in range(num_plots):
                kwargs = {
                    "prior": self.sim_model.prior,
                    "posterior_draws": self.test_dict["posterior_samples_unnorm"][i],
                }
                self.plot_wrapper(
                    diag.plot_posterior_2d,
                    parent_folder,
                    "prior and posterior comparison",
                    "compare_prior_post_{}.png".format(i),
                    **kwargs,
                )

        if self.eval_settings["plot_resimulation"]:
            print("Ploting resimulation...")
            self.sim_model.plot_resimulation(
                self.test_dict["posterior_samples_unnorm"],
                self.test_dict["test_data_raw"]["sim_data"],
                parent_folder,
            )

    def plot_posterior_2d(
        self,
        posterior_draws,
        prior=None,
        prior_draws=None,
        ground_truth=None,
        show_mean=False,
        param_names=None,
        height=2,
        post_color="#8f2727",
        prior_color="gray",
        post_alpha=0.9,
        prior_alpha=0.7,
    ):
        """Generates a bivariate pairplot given posterior draws and prior.

        posterior_draws   : np.ndarray of shape (n_post_draws, n_params)
            The posterior draws obtained for a SINGLE observed data set.
        prior             : bayesflow.forward_inference.Prior instance or None, optional (default: None)
            The optional prior object having an input-output signature as given by ayesflow.forward_inference.Prior
        prior_draws       : np.ndarray of shape (n_prior_draws, n_params) or None, optonal (default: None)
            The optional prior draws obtained from the prior. If both prior and prior_draws are provided, prior_draws
            will be used.
        param_names       : list or None, optional, default: None
            The parameter names for nice plot titles. Inferred if None
        height            : float, optional, default: 2.
            The height of the pairplot.
        post_color        : str, optional, default: '#8f2727'
            The color for the posterior histograms and KDEs.
        priors_color      : str, optional, default: gray
            The color for the optional prior histograms and KDEs.
        post_alpha        : float in [0, 1], optonal, default: 0.9
            The opacity of the posterior plots.
        prior_alpha       : float in [0, 1], optonal, default: 0.7
            The opacity of the prior plots.

        Returns
        -------
        f : plt.Figure - the figure instance for optional saving
        """

        # Ensure correct shape
        assert (
            len(posterior_draws.shape)
        ) == 2, "Shape of `posterior_samples` for a single data set should be 2 dimensional!"

        # Obtain n_draws and n_params
        n_draws, n_params = posterior_draws.shape

        # Determine if prior should be plotted

        # If prior object is given and no draws, obtain draws
        if prior is not None and prior_draws is None:
            draws = prior(n_draws)
            if type(draws) is dict:
                prior_draws = draws["prior_draws"]
            else:
                prior_draws = draws
        # Otherwise, keep as is (prior_draws either filled or None)
        else:
            pass

        # Attempt to determine parameter names
        if param_names is None:
            if hasattr(prior, "param_names"):
                param_names = prior.param_names
            else:
                param_names = [f"Param. {p}" for p in range(1, n_params + 1)]

        # Pack posterior draws into a dataframe
        posterior_draws_df = pd.DataFrame(posterior_draws, columns=param_names)

        # Add posterior
        g = sns.PairGrid(posterior_draws_df, height=height)
        g.map_diag(
            sns.histplot, fill=True, color=post_color, alpha=post_alpha, kde=True
        )
        g.map_lower(sns.kdeplot, fill=True, color=post_color, alpha=post_alpha)

        # Add prior, if given
        if prior_draws is not None:
            prior_draws_df = pd.DataFrame(prior_draws, columns=param_names)
            g.data = prior_draws_df
            g.map_diag(
                sns.histplot,
                fill=True,
                color=prior_color,
                alpha=prior_alpha,
                kde=True,
                zorder=-1,
            )
            g.map_lower(
                sns.kdeplot, fill=True, color=prior_color, alpha=prior_alpha, zorder=-1
            )

        if ground_truth is not None:
            ground_truth_df = pd.DataFrame(ground_truth, columns=param_names)
            g.data = ground_truth_df
            g.map_lower(plt.scatter, alpha=1, s=40, color="blue", marker="x")

        if show_mean:
            mean_pred_df = pd.DataFrame(
                np.expand_dims(np.mean(posterior_draws, axis=0), axis=0),
                columns=param_names,
            )
            g.data = mean_pred_df
            g.map_lower(plt.scatter, alpha=1, s=40, color="green", marker="x")

        # Remove upper axis
        for i, j in zip(*np.triu_indices_from(g.axes, 1)):
            g.axes[i, j].axis("off")

        # Add grids
        for i in range(n_params):
            for j in range(n_params):
                g.axes[i, j].grid(alpha=0.5)

        # Add legend, if prior also given
        if prior_draws is not None:
            handles = [
                Line2D(
                    xdata=[],
                    ydata=[],
                    color=post_color,
                    lw=3,
                    alpha=post_alpha,
                    label="Posterior",
                ),
                Line2D(
                    xdata=[],
                    ydata=[],
                    color=prior_color,
                    lw=3,
                    alpha=prior_alpha,
                    label="Prior",
                ),
            ]
            g.add_legend(handles)
        g.tight_layout()
        return g.fig
