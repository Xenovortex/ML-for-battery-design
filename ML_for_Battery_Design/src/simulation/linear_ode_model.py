import numpy as np
import numpy.typing as npt

from ML_for_Battery_Design.src.simulation.simulation_model import SimulationModel


class LinearODEsystem(SimulationModel):
    """Simulation model represented by a system of two linear ODEs

        du_dt = a * u + b * v
        dv_dt = c * u + d * v

    Attributes:
        # TODO
    """

    def __init__(
        self,
        hidden_params: dict,
        simulation_settings: dict,
        sample_boundaries: dict,
        default_param_values: dict,
    ) -> None:
        """Initializes a :class:LinearODEsystem simulation model

        Args:
            hidden_params (dict): boolean values for if each hidden parameter should be sampled or stay constant
            simulation_settings (dict): settings for generating simulation data
            sample_boundaries (dict): sampling boundaries for each hidden parameter
            default_param_values (dict): default values of hidden parameters, if not sampled
        """
        super().__init__(
            hidden_params, simulation_settings, sample_boundaries, default_param_values
        )
        self.num_features = 2
        self.print_internal_settings()

    def get_sim_data_dim(self) -> tuple:
        """Return dimensions of simulation data

        Returns:
            sim_data_dim (tuple): dimensions of simulatoin data (time points, features)
        """
        sim_data_dim = (self.max_time_iter, self.num_features)
        return sim_data_dim

    def reject_sampler(self, sample: npt.NDArray[np.float64]) -> bool:
        """Reject sample if it will lead to unstable solutions

        Args:
            sample (npt.NDArray[np.float64]): uniform prior sample

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
        if np.any(eigenvalues.real > 0):
            return True
        else:
            return False

    def simulator(self):
        pass

    def plot_sim_data(self):
        pass
