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
        dt0 (float): time step size for discretization in time direction
        max_time_iter (int): number of iterations after which the simulation stops
        nr (int): number of discretization points in space dimension (only for PDE)
        is_pde (bool): if the simulation model is described by PDEs or ODEs
        t (npt.NDArray[Any]): time points at which the solutions should be evaluated
        hidden_param_names (list): list of hidden parameter names
        default_param_kwargs (dict): not-sampled parameters default values as keyword arguments
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
                "{}: hidden_params input is not dictionary type".format(
                    self.__class__.__name__
                )
            )
        if not isinstance(self.simulation_settings, dict):
            raise TypeError(
                "{}: simulation_settings input is not dictionary type".format(
                    self.__class__.__name__
                )
            )
        if not isinstance(self.sample_boundaries, dict):
            raise TypeError(
                "{}: sample_boundaries input is not dictionary type".format(
                    self.__class__.__name__
                )
            )
        if not isinstance(self.default_param_values, dict):
            raise TypeError(
                "{}: default_param_values input is not dictionary type".format(
                    self.__class__.__name__
                )
            )

        # unpack simulation parameters
        self.dt0 = self.simulation_settings["dt0"]
        self.max_time_iter = self.simulation_settings["max_time_iter"]
        if "nr" in simulation_settings:
            self.nr = self.simulation_settings["nr"]
            self.is_pde = True
        else:
            self.is_pde = False

        # prepare simulation model
        self.t = self.get_time_points()
        self.hidden_param_names = self.get_param_names()
        self.default_param_kwargs = self.get_default_param_kwargs()

        # warning, if no hidden parameters
        if len(self.hidden_param_names) == 0:
            print(
                "Warning: {} - No hidden parameters to sample.".format(
                    self.__class__.__name__
                )
            )

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
            t (npt.NDArray[Any]): time points corresponding to simulation data
        """
        t = np.linspace(0, (self.max_time_iter - 1) * self.dt0, num=self.max_time_iter)
        return t

    def get_param_names(self) -> list:
        """Return list of names for sampled hidden parameters

        Returns:
            hidden_param_names (list): list of hidden parameter names
        """
        hidden_param_names = []
        for param_name in self.default_param_values.keys():
            if self.hidden_params["sample_" + param_name]:
                hidden_param_names.append(param_name)
        return hidden_param_names

    def get_default_param_kwargs(self) -> dict:
        """Return default constant values of not-sampled parameters as keyword arguments

        Returns:
            default_param_kwargs (dict): not-sampled parameters default values as keyword arguments
        """
        default_param_kwargs = {}
        for param_name in self.default_param_values.keys():
            if not self.hidden_params["sample_" + param_name]:
                default_param_kwargs[param_name] = self.default_param_values[param_name]
        return default_param_kwargs

    def print_internal_settings(self) -> None:
        """Print internal simulation settings to console"""
        print("hidden parameters: {}".format(self.hidden_param_names))
        print("dt0: {}".format(self.dt0))
        print("max_time_iter: {}".format(self.max_time_iter))
        if self.is_pde:
            print("nr: {}".format(self.nr))
        print("simulation data dimensions: {}".format(self.get_sim_data_dim()))
        print()
        print("parameter values:")
        for key, value in self.hidden_params.items():
            if value:
                print(
                    "{}: {} -> boundary".format(
                        key[len("sample_") :],
                        self.sample_boundaries[key[len("sample_") :]],
                    )
                )
            else:
                print(
                    "{}: {} -> constant".format(
                        key[len("sample_") :],
                        self.default_param_values[key[len("sample_") :]],
                    )
                )

    def sample_to_kwargs(self, sample: npt.NDArray[np.float32]) -> dict:
        """Convert hidden parameter sample to keyword arguments and add default values for non-sampled parameters

        Args:
            sample (npt.NDArray[np.float32]): hidden parameter prior sample

        Returns:
            param_kwargs (dict): prior samples and default constant non-sampled parameters as keyword arguments
        """
        sample_kwargs = dict(zip(self.hidden_param_names, sample))
        param_kwargs = {**sample_kwargs, **self.default_param_kwargs}
        return param_kwargs

    def reject_sampler(self, sample: npt.NDArray[np.float64]) -> bool:
        """Dummy rejection sampler that accepts all samples (no rejection)

        Args:
            sample (npt.NDArray[np.float32]): hidden parameter prior sample

        Returns:
            bool: if sample should be rejected or not
        """
        return False

    def uniform_prior(self, reject_sampling: bool = False) -> npt.NDArray[np.float32]:
        """Generate samples from uniform prior

        Args:
            reject_sampling (bool, optional): If True, rejection sampling will be performed. Defaults to False.

        Returns:
            sample (npt.NDArray[np.float32]): uniform prior sample
        """
        lower_boundary = []
        upper_boundary = []

        for param_name in self.hidden_param_names:
            lower_boundary.append(self.sample_boundaries[param_name][0])
            upper_boundary.append(self.sample_boundaries[param_name][1])

        if reject_sampling:
            while True:
                sample = np.random.uniform(
                    low=lower_boundary,
                    high=upper_boundary,
                    size=len(self.hidden_param_names),
                )

                if not self.reject_sampler(sample):
                    break
        else:
            sample = np.random.uniform(
                low=lower_boundary,
                high=upper_boundary,
                size=len(self.hidden_param_names),
            )

        return sample.astype(np.float32)
