from typing import Any

import numpy as np
import numpy.typing as npt


class Processing:
    """Processing model for transforming batches during inference

    Attributes:
        processing_settings (dict): settings on which processing transforms should be performed
        prior_means (npt.NDArray[Any]): estimated mean of joint prior for normalization
        prior_stds (npt.NDArray[Any]): estimated standard deviation of joint prior for normalization
        sim_data_means (npt.NDArray[Any]): estimated mean of simulation data for normalization
        sim_data_stds (npt.NDArray[Any]): estimated standard deviation of simulation data for normalization
    """

    def __init__(
        self,
        processing_settings: dict,
        prior_means: npt.NDArray[Any] = None,
        prior_stds: npt.NDArray[Any] = None,
        sim_data_means: npt.NDArray[Any] = None,
        sim_data_stds: npt.NDArray[Any] = None,
    ) -> None:
        """Initializes :class:Processing model

        Args:
            processing_settings (dict): settings on which processing transforms should be performed
            prior_means (npt.NDArray[Any], optional): estimated mean of joint prior for normalization. Defaults to None.
            prior_stds (npt.NDArray[Any], optional): estimated standard deviation of joint prior for normalization. Defaults to None.
            sim_data_means (npt.NDArray[Any], optional): estimated mean of simulation data for normalization. Defaults to None.
            sim_data_stds (npt.NDArray[Any], optional): estimated standard deviation of simulation data for normalization. Defaults to None.
        """
        self.settings = processing_settings
        self.prior_means = prior_means
        self.prior_stds = prior_stds
        self.sim_data_means = sim_data_means
        self.sim_data_stds = sim_data_stds

        if not isinstance(self.settings, dict):
            raise TypeError(
                "{} - init: processing_settings input is not dictionary type".format(
                    self.__class__.__name__
                )
            )

        if self.settings["norm_prior"]:
            if self.prior_means is None:
                raise ValueError(
                    "{} - init: processing setting norm_prior is True, but prior_means is None.".format(
                        self.__class__.__name__
                    )
                )
            if self.prior_stds is None:
                raise ValueError(
                    "{} - init: processing setting norm_prior is True, but prior_stds is None".format(
                        self.__class__.__name__
                    )
                )

        if self.settings["norm_sim_data"] == "mean_std":
            if self.sim_data_means is None:
                raise ValueError(
                    "{} - init: processing setting norm_sim_data is mean_std, but sim_data_means is None".format(
                        self.__class__.__name__
                    )
                )
            if self.sim_data_stds is None:
                raise ValueError(
                    "{} - init: processing setting norm_sim_data is mean_std, but sim_data_stds is None".format(
                        self.__class__.__name__
                    )
                )

    def call(self, forward_dict: dict) -> dict:
        """ "Perform processing transform on input batch for BayesFlow

        Args:
            forward_dict (dict): batch of priors and simulation data returned from BayesFlow GenerativeModel class

        Returns:
            out_dict (dict): batch of priors and simulation data with processing transformations performed
        """
        out_dict = {}
        params = forward_dict["prior_draws"]
        sim_data = forward_dict["sim_data"]

        if self.settings["norm_prior"]:
            params = (params - self.prior_means) / self.prior_stds

        if self.settings["norm_sim_data"] is not None:
            if self.settings["norm_sim_data"] == "log_norm":
                sim_data = np.log1p(sim_data)
            if self.settings["norm_sim_data"] == "mean_std":
                sim_data = (sim_data - self.sim_data_means) / self.sim_data_stds
            else:
                raise ValueError(
                    "{} - call: processing setting norm_sim_data '{}' is not a valid input".format(
                        self.__class__.__name__, self.settings["norm_sim_data"]
                    )
                )

        if self.settings["remove_nan"]:
            if sim_data.ndim == 3:
                keep_idx = np.all(np.isfinite(sim_data), axis=(1, 2))
            elif sim_data.ndim == 4:
                keep_idx = np.all(np.isfinite(sim_data), axis=(1, 2, 3))
            if not np.all(keep_idx):
                print("Invalid value encountered...removing from batch")
            params = params[keep_idx]
            sim_data = sim_data[keep_idx]

        out_dict["parameters"] = params
        out_dict["summary_conditions"] = sim_data

        return out_dict
