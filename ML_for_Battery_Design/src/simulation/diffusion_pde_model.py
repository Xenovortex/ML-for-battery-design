import os
import pathlib
import time
from typing import Any, Tuple, Type

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.integrate import odeint

from ML_for_Battery_Design.src.simulation.simulation_model import SimulationModel


class DiffusionPDEModel(SimulationModel):
    def __init__(
        self,
        hidden_params: dict,
        simulation_settings: dict,
        sample_boundaries: dict,
        default_param_values: dict,
        plot_settings: dict,
    ) -> None:
        self.num_features = 2
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
        sim_data_dim = (self.max_time_iter, self.nr, self.num_features)
        return sim_data_dim

    def solver(self, params: npt.NDArray[Any]) -> npt.NDArray[Any]:
        param_kwargs = self.sample_to_kwargs(params)
        X0 = np.ones(2 * self.nr)

        def diffusion_pde(X, t):
            diff_matrix = (
                np.diag(np.full(self.nr, -2))
                + np.diag(np.ones(self.nr - 1), 1)
                + np.diag(np.ones(self.nr - 1), -1)
                + np.diag(np.ones(1), self.nr - 1)
                + np.diag(np.ones(1), 1 - self.nr)
            )

            zero_matrix = np.zeros((self.nr, self.nr))

            Du_matrix = param_kwargs["Du"] * diff_matrix
            Dv_matrix = param_kwargs["Dv"] * diff_matrix

            Du_matrix_zero = np.concatenate((Du_matrix, zero_matrix), axis=1)
            Dv_matrix_zero = np.concatenate((zero_matrix, Dv_matrix), axis=1)

            D_matrix = np.concatenate((Du_matrix_zero, Dv_matrix_zero), axis=0)

            a_diag_matrix = np.diag(np.full(self.nr, param_kwargs["a"]))
            b_diag_matrix = np.diag(np.full(self.nr, param_kwargs["b"]))
            c_diag_matrix = np.diag(np.full(self.nr, param_kwargs["c"]))
            d_diag_matrix = np.diag(np.full(self.nr, param_kwargs["d"]))

            ab_matrix = np.concatenate((a_diag_matrix, b_diag_matrix), axis=1)
            cd_matrix = np.concatenate((c_diag_matrix, d_diag_matrix), axis=1)

            abcd_matrix = np.concatenate((ab_matrix, cd_matrix), axis=0)

            if self.simulation_settings["use_f_terms"]:
                f_u = (
                    param_kwargs["beta_u"]
                    * param_kwargs["gamma_u"]
                    * np.exp(param_kwargs["beta_u"] * t)
                    * np.sin(param_kwargs["alpha_u"] * self.x)
                    - param_kwargs["a"]
                    * param_kwargs["gamma_u"]
                    * np.exp(param_kwargs["beta_u"] * t)
                    * np.sin(param_kwargs["alpha_u"] * self.x)
                    - param_kwargs["b"]
                    * param_kwargs["gamma_v"]
                    * np.exp(param_kwargs["beta_v"] * t)
                    * np.cos(param_kwargs["alpha_v"] * self.x)
                    + param_kwargs["Du"]
                    * param_kwargs["alpha_u"] ** 2
                    * param_kwargs["gamma_u"]
                    * np.exp(param_kwargs["beta_u"] * t)
                    * np.sin(param_kwargs["alpha_u"] * self.x)
                )

                f_v = (
                    param_kwargs["beta_v"]
                    * param_kwargs["gamma_v"]
                    * np.exp(param_kwargs["beta_v"] * t)
                    * np.cos(param_kwargs["alpha_v"] * self.x)
                    - param_kwargs["d"]
                    * param_kwargs["gamma_v"]
                    * np.exp(param_kwargs["beta_v"] * t)
                    * np.cos(param_kwargs["alpha_v"] * self.x)
                    - param_kwargs["c"]
                    * param_kwargs["gamma_u"]
                    * np.exp(param_kwargs["beta_u"] * t)
                    * np.sin(param_kwargs["alpha_u"] * self.x)
                    + param_kwargs["Dv"]
                    * param_kwargs["alpha_v"] ** 2
                    * param_kwargs["gamma_v"]
                    * np.exp(param_kwargs["beta_v"] * t)
                    * np.cos(param_kwargs["alpha_v"] * self.x)
                )

                f = np.concatenate((f_u, f_v), axis=0)
                dX_dt = (D_matrix + abcd_matrix) @ X + f
            else:
                dX_dt = (D_matrix + abcd_matrix) @ X

            return dX_dt

        solution = odeint(
            diffusion_pde,
            X0,
            self.t,
            rtol=self.simulation_settings["rtol"],
            atol=self.simulation_settings["atol"],
            hmin=self.simulation_settings["hmin"],
        )

        u = np.expand_dims(solution[:, : self.nr], axis=2)
        v = np.expand_dims(solution[:, self.nr :], axis=2)

        sim_data = np.concatenate((u, v), axis=2)

        return sim_data

    def plot_sim_data(
        self, parent_folder: str = None
    ) -> Tuple[Type[Figure], Type[Axes], npt.NDArray[Any], npt.NDArray[Any]]:
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

        # plot u
        for i in range(self.plot_settings["num_plots"]):
            ax = fig.add_subplot(
                int("{}{}{}".format(n_row, n_col, i + 1)), projection="3d"
            )
            nr = np.linspace(1, self.nr, self.nr)
            X, Y = np.meshgrid(nr, self.t)
            mappable = plt.cm.ScalarMappable()
            mappable.set_array(sim_data[i, :, :, 0])
            ax.plot_surface(
                X,
                Y,
                sim_data[i, :, :, 0],
                cmap=mappable.cmap,
                norm=mappable.norm,
                linewidth=0,
                antialiased=False,
                alpha=0.8,
                label="asynchronous",
            )
            fig.colorbar(mappable)

            ax.set_xlabel("Position x [m]")
            ax.set_ylabel("Time t [s]")
            ax.set_zlabel("Diffusion u(x,t)")
            ax.grid(True)

        if self.plot_settings["show_title"]:
            fig.suptitle("Diffusion PDE model simulation data u(x,t) examples")

        plt.tight_layout()

        if parent_folder is not None:
            pathlib.Path(parent_folder).mkdir(parents=True, exist_ok=True)
            save_path = os.path.join(parent_folder, "sim_data_u.png")
            fig.savefig(save_path, transparent=True, bbox_inches="tight", pad_inches=0)

        if self.plot_settings["show_plot"]:
            plt.show(block=True if self.plot_settings["show_time"] is None else False)
            if self.plot_settings["show_time"] is not None:
                time.sleep(self.plot_settings["show_time"])
        plt.close()

        fig = plt.figure(figsize=self.plot_settings["figsize"])

        # plot v
        for i in range(self.plot_settings["num_plots"]):
            ax = fig.add_subplot(
                int("{}{}{}".format(n_row, n_col, i + 1)), projection="3d"
            )
            X, Y = np.meshgrid(self.x, self.t)
            mappable = plt.cm.ScalarMappable()
            mappable.set_array(sim_data[i, :, :, 1])
            ax.plot_surface(
                X,
                Y,
                sim_data[i, :, :, 1],
                cmap=mappable.cmap,
                norm=mappable.norm,
                linewidth=0,
                antialiased=False,
                alpha=0.8,
                label="asynchronous",
            )
            fig.colorbar(mappable)

            ax.set_xlabel("Position x [m]")
            ax.set_ylabel("Time t [s]")
            ax.set_zlabel("Diffusion v(x,t)")
            ax.grid(True)

        if self.plot_settings["show_title"]:
            fig.suptitle("Diffusion PDE model simulation data v(x,t) examples")

        plt.tight_layout()

        if parent_folder is not None:
            pathlib.Path(parent_folder).mkdir(parents=True, exist_ok=True)
            save_path = os.path.join(parent_folder, "sim_data_v.png")
            fig.savefig(save_path, transparent=True, bbox_inches="tight", pad_inches=0)

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
    ) -> Tuple[Type[Figure], Type[Axes], npt.NDArray[Any]]:

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

        for k in range(self.nr):
            fig = plt.figure(figsize=self.plot_settings["figsize"])
            for i in range(self.plot_settings["num_plots"]):
                ax = fig.add_subplot(int("{}{}{}".format(n_row, n_col, i + 1)))
                ax.plot(
                    self.t,
                    np.median(resim[i, :, :, k, 0], axis=0),
                    label="Median u({},t)".format(k),
                )
                ax.plot(
                    self.t,
                    ground_truths[i, :, k, 0],
                    marker="o",
                    label="Ground truth u({},t)".format(k),
                    color="k",
                    linestyle="--",
                    alpha=0.8,
                )

                qt_50 = np.quantile(resim[i, :, :, k, 0], q=[0.25, 0.75], axis=0)
                qt_90 = np.quantile(resim[i, :, :, k, 0], q=[0.05, 0.95], axis=0)
                qt_95 = np.quantile(resim[i, :, :, k, 0], q=[0.025, 0.975], axis=0)

                ax.fill_between(self.t, qt_50[0], qt_50[1], alpha=0.3, label="50% CI")
                ax.fill_between(self.t, qt_90[0], qt_90[1], alpha=0.2, label="90% CI")
                ax.fill_between(self.t, qt_95[0], qt_95[1], alpha=0.1, label="95% CI")

                ax.set_xlabel("Time [s]")
                ax.set_ylabel("Diffusion u({},t)".format(k))
                ax.grid(True)
                handles, labels = ax.get_legend_handles_labels()

            fig.legend(handles, labels)

            if self.plot_settings["show_title"]:
                fig.suptitle("Diffusion PDE model resimulation u({}, t)".format(k))

            plt.tight_layout()

            if parent_folder is not None:
                pathlib.Path(parent_folder).mkdir(parents=True, exist_ok=True)
                save_path = os.path.join(
                    parent_folder, "resimulation_u_nr{}.png".format(k)
                )
                fig.savefig(
                    save_path, transparent=True, bbox_inches="tight", pad_inches=0
                )

            if self.plot_settings["show_plot"]:
                plt.show(
                    block=True if self.plot_settings["show_time"] is None else False
                )
                if self.plot_settings["show_time"] is not None:
                    time.sleep(self.plot_settings["show_time"])
            plt.close()

        for k in range(self.nr):
            fig = plt.figure(figsize=self.plot_settings["figsize"])
            for i in range(self.plot_settings["num_plots"]):
                ax = fig.add_subplot(int("{}{}{}".format(n_row, n_col, i + 1)))
                ax.plot(
                    self.t,
                    np.median(resim[i, :, :, k, 1], axis=0),
                    label="Median v({},t)".format(k),
                )
                ax.plot(
                    self.t,
                    ground_truths[i, :, k, 1],
                    marker="o",
                    label="Ground truth v({}, t)".format(k),
                    color="k",
                    linestyle="--",
                    alpha=0.8,
                )

                qt_50 = np.quantile(resim[i, :, :, k, 0], q=[0.25, 0.75], axis=0)
                qt_90 = np.quantile(resim[i, :, :, k, 0], q=[0.05, 0.95], axis=0)
                qt_95 = np.quantile(resim[i, :, :, k, 0], q=[0.025, 0.975], axis=0)

                ax.fill_between(self.t, qt_50[0], qt_50[1], alpha=0.3, label="50% CI")
                ax.fill_between(self.t, qt_90[0], qt_90[1], alpha=0.2, label="90% CI")
                ax.fill_between(self.t, qt_95[0], qt_95[1], alpha=0.1, label="95% CI")

                ax.set_xlabel("Time [s]")
                ax.set_ylabel("Diffusion v({},t)".format(k))
                ax.grid(True)
                handles, labels = ax.get_legend_handles_labels()

            fig.legend(handles, labels)

            if self.plot_settings["show_title"]:
                fig.suptitle("Diffusion PDE model resimulation v({}, t)".format(k))

            plt.tight_layout()

            if parent_folder is not None:
                pathlib.Path(parent_folder).mkdir(parents=True, exist_ok=True)
                save_path = os.path.join(
                    parent_folder, "resimulation_v_nr{}.png".format(k)
                )
                fig.savefig(
                    save_path, transparent=True, bbox_inches="tight", pad_inches=0
                )

            if self.plot_settings["show_plot"]:
                plt.show(
                    block=True if self.plot_settings["show_time"] is None else False
                )
                if self.plot_settings["show_time"] is not None:
                    time.sleep(self.plot_settings["show_time"])
            plt.close()

        return fig, ax, resim
