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
        self.num_features = 4
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

    def simulator(self, params: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Returns analytical solutions u and v of linear ODE system

        Args:
           params (npt.NDArray[np.float32]): hidden prior paramters

        Returns:
            solution (npt.NDArray[np.float32]): ODE solutions of size (time points, 4) with entries (u(t).real, v(t).real, u(t).imag, v(t).imag) at each time point t
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
        solution = np.concatenate((solution.T.real, solution.T.imag), axis=1)
        return solution.astype(np.float32)

    def plot_sim_data(self):
        pass
