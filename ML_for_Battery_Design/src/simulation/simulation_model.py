from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt


class SimulationModel(ABC):
    """Simulation model abstract class for hidden parameter sampling and simulation data generation

    Attributes:
        hidden_params (dict): boolean values for if each hidden parameter should be sampled or stay constant
        simulation_settings (dict): settings for generating simulation data
        sample_boundaries (dict): sampling boundaries for each hidden parameter
        default_param_values (dict): default values of hidden parameters, if not sampled
        dt0 (float):
    """

    def __init__(
        self,
        hidden_params: dict,
        simulation_settings: dict,
        sample_boundaries: dict,
        default_param_values: dict,
    ) -> None:
        """Initializes parent :class:SimulationModel

        Args:
            hidden_params (dict): boolean values for if each hidden parameter should be sampled or stay constant
            simulation_settings (dict): settings for generating simulation data
            sample_boundaries (dict): sampling boundaries for each hidden parameter
            default_param_values (dict): default values of hidden parameters, if not sampled
        """
        self.hidden_params = hidden_params
        self.simulation_settings = simulation_settings
        self.sample_boundaries = sample_boundaries
        self.default_param_values = default_param_values

        # input type check
        if not isinstance(self.hidden_params, dict):
            raise TypeError(
                "SimulationModel: hidden_params input is not dictionary type"
            )
        if not isinstance(self.simulation_settings, dict):
            raise TypeError(
                "SimulationModel: simulation_settings input is not dictionary type"
            )
        if not isinstance(self.sample_boundaries, dict):
            raise TypeError(
                "SimulationModel: sample_boundaries input is not dictionary type"
            )
        if not isinstance(self.default_param_values, dict):
            raise TypeError(
                "SimulationModel: default_param_values input is not dictionary type"
            )

        # unpach simulation parameters
        self.dt0 = self.simulation_settings["dt0"]
        self.max_time_iter = self.simulation_settings["max_time_iter"]
        if "nr" in simulation_settings:
            self.nr = self.simulation_settings["nr"]
            self.is_pde = True
        else:
            self.is_pde = False

    @abstractmethod
    def get_sim_data_dim(self):
        raise NotImplementedError(
            "SimulationModel: get_sim_data_dim method is not implement"
        )

    @abstractmethod
    def simulator(self):
        raise NotImplementedError("SimulationModel: simulator method is not implement")

    @abstractmethod
    def plot_sim_data(self):
        raise NotImplementedError("SimulationModel: plot_sim_data is not implement")

    def get_time_points(self) -> npt.NDArray[Any]:
        """Return time points for generation of simulation data

        Returns:
            t (np.array): time points corresponding to simulation data
        """
        t = np.linspace(0, (self.max_time_iter - 1) * self.dt0, num=self.max_time_iter)
        return t
