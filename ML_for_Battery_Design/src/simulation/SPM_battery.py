import contextlib
import os
import pathlib
import time
from typing import Any, Tuple, Type

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from SPM1.Electrode_V0 import Electrode

from ML_for_Battery_Design.src.simulation.simulation_model import SimulationModel


class SPMBatteryModel(SimulationModel):
    def __init__(
        self,
        hidden_params: dict,
        simulation_settings: dict,
        sample_boundaries: dict,
        default_param_values: dict,
        plot_settings: dict,
    ) -> None:
        self.num_features = 1
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

        printing = False
        save_output = False
        stop_condition = 1
        V_cut = 3
        param_kwargs = self.sample_to_kwargs(params)

        # settings
        consts = {"F": 96487, "R": 8.31446262}
        store_solidDiff_intermediate_solRHS = False
        store_solidDiff_intermediate_solMatrix = (
            False  # solMatrix is unchanged for now!
        )
        intermediateStorage_timeStepFreq = 100  # applicable when a store_*==True

        # assign model paremeters:
        with open(os.devnull, "w") as file, contextlib.redirect_stdout(file):
            an = Electrode(LiPlate=True, k=3.277413185370443e-05)  # k~100
            cat = Electrode(
                L=param_kwargs["L"],
                eps=param_kwargs["eps"],
                r=param_kwargs["r"],
                Ds=param_kwargs["Ds"],
                cs_max=43927,
                soc_0=0.1738,
                soc_max=0.94680575,
                k=param_kwargs["k"],
            )  # k~0.7
        liquidProp = {"ce": 1000}

        # =====================================================================
        if printing:
            print("calculating... \n")

        # calculate reaction current (currently for constant charge/discharge):
        current = cat.i_1C * param_kwargs["C_rate"]
        cat_j = -current / (cat.a * cat.L * consts["F"] * cat.A)
        an_j = current / (consts["F"] * an.A)  # lithium metal anode

        # init output
        output: dict[str, Any] = {
            "an": {"eta": []},
            "cat": {"cs": [[cat.soc_0 * cat.cs_max] * self.nr], "eta": [], "U0": []},
            "V": [],
            "t_seq": [0],
        }

        nt = 0
        while True:
            dt = self.dt0  # constant time steps for now!

            # solve cs via solid duffusion eq.:
            if nt != 0:
                output["cat"]["cs"].append(
                    cat.solveDiff(dt, self.nr, cat_j, output["cat"]["cs"][nt - 1])
                )

            # calculate overpotentials via BV eq.:
            output["cat"]["eta"].append(
                2
                * consts["R"]
                * cat.T
                / consts["F"]
                * np.arcsinh(
                    0.5
                    * cat_j
                    * consts["F"]
                    / (
                        cat.k
                        * consts["F"]
                        * pow(liquidProp["ce"], 0.5)
                        * pow(output["cat"]["cs"][nt][self.nr - 1], 0.5)
                        * pow(
                            max(
                                1e-10,
                                (cat.cs_max - output["cat"]["cs"][nt][self.nr - 1]),
                            ),
                            0.5,
                        )
                    )
                )
            )

            output["an"]["eta"].append(
                2
                * consts["R"]
                * an.T
                / consts["F"]
                * np.arcsinh(
                    0.5
                    * an_j
                    * consts["F"]
                    / (an.k * consts["F"] * pow(liquidProp["ce"], 0.5))
                )
            )  # lithium metal anode

            # calculate cell voltage:
            output["cat"]["U0"].append(
                np.interp(
                    output["cat"]["cs"][nt][self.nr - 1] / cat.cs_max,
                    cat.OCP[:, 0],
                    cat.OCP[:, 1],
                )
            )
            output["V"].append(
                (output["cat"]["U0"][nt] - 0)
                + (output["cat"]["eta"][nt] - output["an"]["eta"][nt])
                + (0)
            )

            # convergence/stop condition check:
            if nt == 1:
                solidDiff_matrix = cat.diffMatrix[:, :, np.newaxis]
                solidDiff_rhs = np.resize(cat.diffRHS, (self.nr, 1))
            elif (nt % intermediateStorage_timeStepFreq) == 1:
                if store_solidDiff_intermediate_solRHS:
                    solidDiff_rhs = np.append(
                        solidDiff_rhs, np.resize(cat.diffRHS, (self.nr, 1)), axis=1
                    )
                if store_solidDiff_intermediate_solMatrix:
                    solidDiff_matrix = np.append(
                        solidDiff_matrix, cat.diffMatrix[:, :, np.newaxis], axis=2
                    )

            # convergence/stop condition check:
            if stop_condition == 1 and nt == self.max_time_iter - 1:
                if printing:
                    print("Maximum time step reached!")
                break
            elif stop_condition == 2 and output["V"][nt] <= V_cut:
                if printing:
                    print("Cutoff voltage reached!")
                break
            elif stop_condition != 1 and stop_condition != 2:
                print("Error: undifined stop_condition!")
                break

            output["t_seq"].append(output["t_seq"][-1] + dt)
            nt = nt + 1

        if printing:
            print("\n----------------------- Done! ----------------------")

        if save_output:
            with open("Results.txt", "w") as writer:
                writer.write(
                    "t[s] \t SOC[%] \t cell_Voltage[V] \t  cat_overpotential[V] \n"
                )
                for i in range(len(output["t_seq"])):
                    writer.write(
                        format(output["t_seq"][i])
                        + "\t"
                        + format(
                            100 * np.average(output["cat"]["cs"][i][:]) / cat.cs_max
                        )
                        + "\t"
                        + format(output["V"][i])
                        + "\t"
                        + format(output["cat"]["eta"][i])
                        + "\n"
                    )
                writer.close()

        sim_data = np.expand_dims(np.array(output["cat"]["cs"]), axis=2)
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
            ax.set_zlabel("cs [mol/m^3]")
            ax.grid(True)

        if self.plot_settings["show_title"]:
            fig.suptitle("SPM Battery model simulation data examples")

        plt.tight_layout()

        if parent_folder is not None:
            pathlib.Path(parent_folder).mkdir(parents=True, exist_ok=True)
            save_path = os.path.join(parent_folder, "sim_data.png")
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

        for k in range(self.nr):
            fig = plt.figure(figsize=self.plot_settings["figsize"])
            for i in range(self.plot_settings["num_plots"]):
                ax = fig.add_subplot(int("{}{}{}".format(n_row, n_col, i + 1)))
                ax.plot(self.t, np.median(resim[i, :, :, k, 0], axis=0), label="Median")
                ax.plot(
                    self.t,
                    ground_truths[i, :, k, 0],
                    marker="o",
                    label="Ground truth",
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
                ax.set_ylabel("Concentration [mol/m³]")
                ax.grid(True)
                handles, labels = ax.get_legend_handles_labels()

            fig.legend(handles, labels)

            if self.plot_settings["show_title"]:
                fig.suptitle("SPM Battery model resimulation nr={}".format(k))

            plt.tight_layout()

            if parent_folder is not None:
                pathlib.Path(parent_folder).mkdir(parents=True, exist_ok=True)
                save_path = os.path.join(
                    parent_folder, "resimulation_nr{}.png".format(k)
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
