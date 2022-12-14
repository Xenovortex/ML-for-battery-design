import os
import pathlib
import time
from typing import Any, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ML_for_Battery_Design.src.simulation.simulation_model import SimulationModel


class LinearODEsystem(SimulationModel):
    """Simulation model represented by a system of two linear ODEs

        du_dt = a * u + b * v
        dv_dt = c * u + d * v

    Attributes:
        hidden_params (dict): boolean values for if each hidden parameter should be sampled or stay constant
        simulation_settings (dict): settings for generating simulation data
        sample_boundaries (dict): sampling boundaries for each hidden parameter
        default_param_values (dict): default values of hidden parameters, if not sampled
        plot_settings (dict): settings for plotting simulation data
        dt0 (float): time step size for discretization in time direction
        max_time_iter (int): number of iterations after which the simulation stops
        nr (int): number of discretization points in space dimension (only for PDE)
        reject_sampling (bool): If True, rejection sampling will be performed
        is_pde (bool): if the simulation model is described by PDEs or ODEs
        t (npt.NDArray[Any]): time points at which the solutions should be evaluated
        hidden_param_names (list): list of hidden parameter names
        num_hidden_params (int): number of hidden parameters
        default_param_kwargs (dict): not-sampled parameters default values as keyword arguments
        use_complex (bool): If True, consider complex part of simulation data
        num_features (int): number of features in simulation data per time point
        prior (bayesflow.Prior): prior sample generator wrapped in bayesflow Prior object
        simulator (bayesflow.Simulator): simulation data generator wrapped in bayesflow Simulator object
        generative_model (bayesflow.GenerativeModel): generator for both prior samples and simulation data wrapped in bayesflow GenerativeModel object
        prior_means (npt.NDArray[Any]): estimated mean of joint prior
        prior_stds (npt.NDArray[Any]): estimated standard deviation of joint prior
    """

    def __init__(
        self,
        hidden_params: dict,
        simulation_settings: dict,
        sample_boundaries: dict,
        default_param_values: dict,
        plot_settings: dict,
    ) -> None:
        """Initializes :class:LinearODEsystem simulation model

        Args:
            hidden_params (dict): boolean values for if each hidden parameter should be sampled or stay constant
            simulation_settings (dict): settings for generating simulation data
            sample_boundaries (dict): sampling boundaries for each hidden parameter
            default_param_values (dict): default values of hidden parameters, if not sampled
            plot_settings (dict): settings for plotting simulation data
        """
        self.reject_bounds = {
            "real": simulation_settings["reject_bound_real"],
            "complex": simulation_settings["reject_bound_complex"],
        }
        self.use_complex = simulation_settings["use_complex_part"]
        self.num_features = 4 if self.use_complex else 2

        super().__init__(
            hidden_params,
            simulation_settings,
            sample_boundaries,
            default_param_values,
            plot_settings,
        )

        (
            self.prior,
            self.simulator,
            self.generative_model,
        ) = self.get_bayesflow_generator()

        self.print_internal_settings()

    def get_sim_data_shape(self) -> tuple:
        """Return dimensions of simulation data

        Returns:
            sim_data_dim (tuple): dimensions of simulatoin data (time points, features)
        """
        sim_data_dim = (self.max_time_iter, self.num_features)
        return sim_data_dim

    def reject_sampler(self, sample: npt.NDArray[Any]) -> bool:
        """Reject sample if it will lead to unstable solutions

        Args:
            sample (npt.NDArray[Any]): uniform prior sample

        Returns:
            bool: If sample should be rejected or not
        """
        sample_kwargs = self.sample_to_kwargs(sample)
        A = np.array(
            [
                [sample_kwargs["a"], sample_kwargs["b"]],
                [sample_kwargs["c"], sample_kwargs["d"]],
            ]
        )
        eigenvalues, _ = np.linalg.eig(A)

        if self.reject_bounds["real"] is not None:
            for interval in self.reject_bounds["real"]:
                if np.any(eigenvalues.real >= interval[0]) and np.any(
                    eigenvalues.real <= interval[1]
                ):
                    return True
        elif self.reject_bounds["complex"] is not None:
            if self.reject_bounds["complex"] == "zero":
                if np.any(eigenvalues.imag == 0):
                    return True
            elif self.reject_bounds["complex"] == "non_zero":
                if np.any(eigenvalues.imag != 0):
                    return True

        return False

    def solver(self, params: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Returns analytical solutions u and v of linear ODE system

        Args:
           params (npt.NDArray[Any]): hidden prior paramters

        Returns:
            solution (npt.NDArray[Any]): ODE solutions of size (time points, 4) with entries (u(t).real, v(t).real, u(t).imag, v(t).imag) at each time point t
        """
        param_kwargs = self.sample_to_kwargs(params)
        A = np.array(
            [
                [param_kwargs["a"], param_kwargs["b"]],
                [param_kwargs["c"], param_kwargs["d"]],
            ]
        )
        boundary_conditions = np.array([param_kwargs["u0"], param_kwargs["v0"]])
        eigenvalues, eigenvectors = np.linalg.eig(A)
        C = np.linalg.inv(eigenvectors) @ boundary_conditions
        solution = eigenvectors @ np.array(
            [
                C[0] * np.exp(eigenvalues[0] * self.t),
                C[1] * np.exp(eigenvalues[1] * self.t),
            ]
        )
        if self.use_complex:
            solution = np.concatenate((solution.T.real, solution.T.imag), axis=1)
        else:
            solution = solution.T.real
        return solution

    def plot_sim_data(
        self,
        parent_folder: str = None,
    ) -> Tuple[
        Type[Figure],
        Union[Type[Axes], Type[np.flatiter]],
        npt.NDArray[Any],
        npt.NDArray[Any],
    ]:
        """Generate simulation data plots

        Args:
            parent_folder (str, optional): If given, save plot under parent_folder/sim_data.png. Defaults to None.

        Returns:
            fig (plt.Figure) : matplotlib Figure instance for external access
            ax (plt.Axes) : matplotlib Axes instance for external access
            params (npt.NDArray[Any]): prior samples used to generate the plots
            sim_data (npt.NDArray[Any]): simulation data used to generate the plots
        """
        if self.plot_settings["num_plots"] < 1:
            raise ValueError(
                "{} - plot_sim_data: num_plots is {}, but can not be negative or zero".format(
                    self.__class__.__name__, self.plot_settings["num_plots"]
                )
            )

        plt.rcParams["font.size"] = self.plot_settings["font_size"]

        data_dict = self.generative_model(batch_size=self.plot_settings["num_plots"])
        params = data_dict["prior_draws"]
        sim_data = data_dict["sim_data"]

        n_row = int(np.ceil(len(params) / 6))
        n_col = int(np.ceil(len(params) / n_row))

        fig = plt.figure(figsize=self.plot_settings["figsize"])

        for i in range(self.plot_settings["num_plots"]):
            ax = fig.add_subplot(int("{}{}{}".format(n_row, n_col, i + 1)))
            ax.plot(self.t, sim_data[i, :, 0], label="u(t) real part", c="orange")
            ax.plot(self.t, sim_data[i, :, 1], label="v(t) real part", c="blue")

            if self.use_complex:
                ax.plot(
                    self.t,
                    sim_data[i, :, 2],
                    label="u(t) complex part",
                    c="orange",
                    linestyle="--",
                )
                ax.plot(
                    self.t,
                    sim_data[i, :, 3],
                    label="v(t) complex part",
                    c="blue",
                    linestyle="--",
                )

            if self.plot_settings["show_params"]:
                for j, param_name in enumerate(self.hidden_param_names):
                    ax.text(
                        0.1,
                        0.7 - 0.05 * j,
                        "{}={:.3f}".format(param_name, params[i, j]),
                        horizontalalignment="left",
                        verticalalignment="center",
                        transform=ax.transAxes,
                        size=8,
                    )

            if self.plot_settings["show_eigen"]:
                param_kwargs = self.sample_to_kwargs(params[i])
                A = np.array(
                    [
                        [param_kwargs["a"], param_kwargs["b"]],
                        [param_kwargs["c"], param_kwargs["d"]],
                    ]
                )
                eigenvalues, eigenvectors = np.linalg.eig(A)
                ax.text(
                    0.1,
                    0.7 + 0.2,
                    "Eigenvalue 1 = {:.2f}".format(eigenvalues[0]),
                    horizontalalignment="left",
                    verticalalignment="center",
                    transform=ax.transAxes,
                    size=8,
                )
                ax.text(
                    0.1,
                    0.7 + 0.15,
                    "Eigenvalue 2 = {:.2f}".format(eigenvalues[1]),
                    horizontalalignment="left",
                    verticalalignment="center",
                    transform=ax.transAxes,
                    size=8,
                )
                ax.text(
                    0.1,
                    0.7 + 0.1,
                    "Eigenvector 1 = ({:.2f}, {:.2f})".format(
                        eigenvectors[0, 0], eigenvectors[1, 0]
                    ),
                    horizontalalignment="left",
                    verticalalignment="center",
                    transform=ax.transAxes,
                    size=8,
                )
                ax.text(
                    0.1,
                    0.7 + 0.05,
                    "Eigenvector 2 = ({:.2f}, {:.2f})".format(
                        eigenvectors[0, 1], eigenvectors[1, 1]
                    ),
                    horizontalalignment="left",
                    verticalalignment="center",
                    transform=ax.transAxes,
                    size=8,
                )

            ax.set_xlabel("Time t [s]")
            ax.set_ylabel("Function u(t)/v(t)")
            ax.grid(True)
            handles, labels = ax.get_legend_handles_labels()

        fig.legend(handles, labels)

        if self.plot_settings["show_title"]:
            fig.suptitle("Linear ODE system simulation data examples")

        plt.tight_layout()

        if parent_folder is not None:
            pathlib.Path(parent_folder).mkdir(parents=True, exist_ok=True)
            save_path = os.path.join(parent_folder, "sim_data.png")
            fig.savefig(
                save_path,
                transparent=True,
                bbox_inches="tight",
                pad_inches=0,
            )

        if self.plot_settings["show_plot"]:
            plt.show(block=True if self.plot_settings["show_time"] is None else False)
            if self.plot_settings["show_time"] is not None:
                time.sleep(self.plot_settings["show_time"])
        plt.close()

        return fig, ax, params, sim_data

    def plot_resimulation(
        self,
        post_samples: npt.NDArray[Any],
        ground_truths: npt.NDArray[Any],
        parent_folder: str = None,
    ) -> Tuple[Type[Figure], Union[Type[Axes], Type[np.flatiter]], npt.NDArray[Any]]:
        plt.rcParams["font.size"] = self.plot_settings["font_size"]

        if self.plot_settings["num_plots"] < 1:
            raise ValueError(
                "{} - plot_resimulation: num_plots is {}, but can not be negative or zero".format(
                    self.__class__.__name__, self.plot_settings["num_plots"]
                )
            )

        if self.plot_settings["num_plots"] > post_samples.shape[0]:
            raise ValueError(
                "{} - plot_resimulation: num_plots is {}, but only {} post_samples given".format(
                    self.__class__.__name__,
                    self.plot_settings["num_plots"],
                    post_samples.shape[0],
                )
            )
        else:
            post_samples = post_samples[: self.plot_settings["num_plots"]]

        resim = np.empty(
            tuple(
                [post_samples.shape[0], post_samples.shape[1]]
                + list(self.get_sim_data_shape())
            )
        )

        for i in range(post_samples.shape[0]):
            for j in range(post_samples.shape[1]):
                resim[i, j] = self.solver(post_samples[i, j])

        n_row = int(np.ceil(self.plot_settings["num_plots"] / 6))
        n_col = int(np.ceil(self.plot_settings["num_plots"] / n_row))

        fig = plt.figure(figsize=self.plot_settings["figsize"])

        for i in range(self.plot_settings["num_plots"]):
            ax = fig.add_subplot(int("{}{}{}".format(n_row, n_col, i + 1)))
            ax.plot(
                self.t,
                np.median(resim[i, :, :, 0], axis=0),
                label="Median u(t) real",
                color="orange",
            )
            ax.plot(
                self.t,
                ground_truths[i, :, 0],
                marker="o",
                label="Ground truth u(t) real",
                color="k",
                linestyle="--",
                alpha=0.8,
            )

            u_qt_50 = np.quantile(resim[i, :, :, 0], q=[0.25, 0.75], axis=0)
            u_qt_90 = np.quantile(resim[i, :, :, 0], q=[0.05, 0.95], axis=0)
            u_qt_95 = np.quantile(resim[i, :, :, 0], q=[0.025, 0.975], axis=0)

            ax.fill_between(
                self.t,
                u_qt_50[0],
                u_qt_50[1],
                color="orange",
                alpha=0.3,
                label="u: 50% CI",
            )
            ax.fill_between(
                self.t,
                u_qt_90[0],
                u_qt_90[1],
                color="orange",
                alpha=0.2,
                label="u: 90% CI",
            )
            ax.fill_between(
                self.t,
                u_qt_95[0],
                u_qt_95[1],
                color="orange",
                alpha=0.1,
                label="u: 95% CI",
            )

            ax.plot(
                self.t,
                np.median(resim[i, :, :, 1], axis=0),
                label="Median v(t) real",
                color="blue",
            )
            ax.plot(
                self.t,
                ground_truths[i, :, 1],
                marker="o",
                label="Ground truth v(t) real",
                color="k",
                linestyle="--",
                alpha=0.8,
            )

            v_qt_50 = np.quantile(resim[i, :, :, 1], q=[0.25, 0.75], axis=0)
            v_qt_90 = np.quantile(resim[i, :, :, 1], q=[0.05, 0.95], axis=0)
            v_qt_95 = np.quantile(resim[i, :, :, 1], q=[0.025, 0.975], axis=0)

            ax.fill_between(
                self.t,
                v_qt_50[0],
                v_qt_50[1],
                color="blue",
                alpha=0.3,
                label="v: 50% CI",
            )
            ax.fill_between(
                self.t,
                v_qt_90[0],
                v_qt_90[1],
                color="blue",
                alpha=0.2,
                label="v: 90% CI",
            )
            ax.fill_between(
                self.t,
                v_qt_95[0],
                v_qt_95[1],
                color="blue",
                alpha=0.1,
                label="v: 95% CI",
            )

            if self.use_complex:
                ax.plot(
                    self.t,
                    np.median(resim[i, :, :, 2], axis=0),
                    label="Median u(t) complex",
                    color="orange",
                    linestyle="--",
                )
                ax.plot(
                    self.t,
                    ground_truths[i, :, 2],
                    marker="o",
                    label="Ground truth u(t) complex",
                    color="k",
                    linestyle="--",
                    alpha=0.8,
                )

                u_complex_qt_50 = np.quantile(resim[i, :, :, 2], q=[0.25, 0.75], axis=0)
                u_complex_qt_90 = np.quantile(resim[i, :, :, 2], q=[0.05, 0.95], axis=0)
                u_complex_qt_95 = np.quantile(
                    resim[i, :, :, 2], q=[0.025, 0.975], axis=0
                )

                ax.fill_between(
                    self.t,
                    u_complex_qt_50[0],
                    u_complex_qt_50[1],
                    color="orange",
                    alpha=0.3,
                    label="u complex: 50% CI",
                )
                ax.fill_between(
                    self.t,
                    u_complex_qt_90[0],
                    u_complex_qt_90[1],
                    color="orange",
                    alpha=0.2,
                    label="u complex: 90% CI",
                )
                ax.fill_between(
                    self.t,
                    u_complex_qt_95[0],
                    u_complex_qt_95[1],
                    color="orange",
                    alpha=0.1,
                    label="u complex: 95% CI",
                )

                ax.plot(
                    self.t,
                    np.median(resim[i, :, :, 3], axis=0),
                    label="Median v(t) complex",
                    color="blue",
                    linestyle="--",
                )
                ax.plot(
                    self.t,
                    ground_truths[0, :, 3],
                    marker="o",
                    label="Ground truth v(t) complex",
                    color="k",
                    linestyle="--",
                    alpha=0.8,
                )

                v_complex_qt_50 = np.quantile(resim[i, :, :, 3], q=[0.25, 0.75], axis=0)
                v_complex_qt_90 = np.quantile(resim[i, :, :, 3], q=[0.05, 0.95], axis=0)
                v_complex_qt_95 = np.quantile(
                    resim[i, :, :, 3], q=[0.025, 0.975], axis=0
                )

                ax.fill_between(
                    self.t,
                    v_complex_qt_50[0],
                    v_complex_qt_50[1],
                    color="blue",
                    alpha=0.3,
                    label="v complex: 50% CI",
                )
                ax.fill_between(
                    self.t,
                    v_complex_qt_90[0],
                    v_complex_qt_90[1],
                    color="blue",
                    alpha=0.2,
                    label="v complex: 90% CI",
                )
                ax.fill_between(
                    self.t,
                    v_complex_qt_95[0],
                    v_complex_qt_95[1],
                    color="blue",
                    alpha=0.1,
                    label="v complex: 95% CI",
                )

            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Function u(t)/v(t)")
            ax.grid(True)
            handles, labels = ax.get_legend_handles_labels()

        fig.legend(handles, labels)

        if self.plot_settings["show_title"]:
            fig.suptitle("Linear ODE system resimulation")

        plt.tight_layout()

        if parent_folder is not None:
            pathlib.Path(parent_folder).mkdir(parents=True, exist_ok=True)
            save_path = os.path.join(parent_folder, "resimulation.png")
            fig.savefig(save_path, transparent=True, bbox_inches="tight", pad_inches=0)
        if self.plot_settings["show_plot"]:
            plt.show(block=True if self.plot_settings["show_time"] is None else False)
            if self.plot_settings["show_time"] is not None:
                time.sleep(self.plot_settings["show_time"])
        plt.close()

        return fig, ax, resim
