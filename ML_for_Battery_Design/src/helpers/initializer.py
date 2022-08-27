import os
import pathlib
from typing import Type

import h5py
import numpy as np
import tensorflow as tf
from bayesflow.amortized_inference import AmortizedPosterior
from bayesflow.helper_functions import build_meta_dict
from bayesflow.networks import InvertibleNetwork
from bayesflow.trainers import Trainer
from tqdm.autonotebook import tqdm

from ML_for_Battery_Design.src.helpers.constants import (
    architecture_settings,
    inference_settings,
    sim_model_collection,
    simulation_settings,
    summary_collection,
)
from ML_for_Battery_Design.src.helpers.evaluater import Evaluater
from ML_for_Battery_Design.src.helpers.filemanager import FileManager
from ML_for_Battery_Design.src.helpers.processing import Processing
from ML_for_Battery_Design.src.simulation.simulation_model import SimulationModel


class Initializer:
    def __init__(self, **kwargs: str) -> None:
        self.sim_model_name = kwargs["<sim_model>"]
        self.summary_net_name = kwargs["<summary_net>"]
        self.data_name = kwargs["<data_name>"]
        self.filename = kwargs["<filename>"]
        self.save_model = bool(kwargs["--save_model"])

        self.sim_model = self.get_sim_model()

        if bool(kwargs["train_online"]):
            self.mode = "train_online"
            self.data_name = "online"
            self.file_manager = FileManager(self.mode, **kwargs)
            self.trainer = self.get_trainer()
        elif bool(kwargs["train_offline"]):
            self.mode = "train_offline"
            self.file_manager = FileManager(self.mode, **kwargs)
            self.trainer = self.get_trainer()
        elif bool(kwargs["generate_data"]):
            self.mode = "generate_data"
            self.file_manager = FileManager(self.mode, **kwargs)
        elif bool(kwargs["analyze_sim"]):
            self.mode = "analyze_sim"
            self.file_manager = FileManager(self.mode, **kwargs)
        elif bool(kwargs["evaluate"]):
            self.mode = "evaluate"
            self.file_manager = FileManager(self.mode, **kwargs)

    def get_sim_model(self) -> Type[SimulationModel]:
        if self.sim_model_name in simulation_settings:
            sim_settings = simulation_settings[self.sim_model_name]
        else:
            raise ValueError(
                "{} - get_sim_model: {} not found in simulation settings".format(
                    self.__class__.__name__, self.sim_model_name
                )
            )
        if self.sim_model_name in sim_model_collection:
            sim_model = sim_model_collection[self.sim_model_name](**sim_settings)
        else:
            raise ValueError(
                "{} - get_sim_model: {} is not a valid simulation model".format(
                    self.__class__.__name__, self.sim_model_name
                )
            )
        return sim_model

    def get_summary_net(self) -> Type[tf.keras.Model]:
        if self.sim_model_name in architecture_settings:
            if self.summary_net_name in architecture_settings[self.sim_model_name]:
                summary_architecture = architecture_settings[self.sim_model_name][
                    self.summary_net_name
                ]
            else:
                raise ValueError(
                    "{} - get_summary_net: {} not found in architecture settings for {} simulation model".format(
                        self.__class__.__name__,
                        self.summary_net_name,
                        self.sim_model_name,
                    )
                )
        else:
            raise ValueError(
                "{} - get_summary_net: {} is not a valid simulation model".format(
                    self.__class__.__name__, self.sim_model_name
                )
            )
        meta_dict = build_meta_dict({}, summary_architecture)
        if self.summary_net_name in summary_collection:
            summary_net = summary_collection[self.summary_net_name](meta_dict)
        else:
            raise ValueError(
                "{} - get_summary_net: {} is not a valid summary network architecture".format(
                    self.__class__.__name__, self.summary_net_name
                )
            )
        return summary_net

    def get_inference_net(self) -> Type[InvertibleNetwork]:
        num_params = self.sim_model.num_hidden_params
        if self.sim_model_name in architecture_settings:
            if "INN" in architecture_settings[self.sim_model_name]:
                inference_architecture = architecture_settings[self.sim_model_name][
                    "INN"
                ]
            else:
                raise ValueError(
                    "{} - get_inference_net: INN not found in architecture settings for {} simulation model".format(
                        self.__class__.__name__, self.sim_model_name
                    )
                )
        else:
            raise ValueError(
                "{} - get_inference_net: {} is not a valid simulation model".format(
                    self.__class__.__name__, self.sim_model_name
                )
            )
        inference_net = InvertibleNetwork(
            {**{"n_params": num_params}, **inference_architecture}
        )
        return inference_net

    def get_amortizer(self) -> Type[AmortizedPosterior]:
        summary_net = self.get_summary_net()
        inference_net = self.get_inference_net()
        amortizer = AmortizedPosterior(
            inference_net, summary_net, name="{}_{}_amortizer"
        )
        return amortizer

    def get_configurator(self) -> Type[Processing]:
        if self.sim_model_name in inference_settings:
            if "processing" in inference_settings[self.sim_model_name]:
                configuator = Processing(
                    inference_settings[self.sim_model_name]["processing"],
                    self.sim_model.prior_means,
                    self.sim_model.prior_stds,
                )
            else:
                raise ValueError(
                    "{} - get_trainer: processing not found in inference settings for {} simulation model".format(
                        self.__class__.__name__, self.sim_model_name
                    )
                )
        else:
            raise ValueError(
                "{} - get_trainer: {} is not a valid simulation model".format(
                    self.__class__.__name__, self.sim_model_name
                )
            )
        return configuator

    def get_trainer(self) -> Type[Trainer]:
        amortizer = self.get_amortizer()
        configurator = self.get_configurator()
        save_model_path = self.file_manager("model") if self.save_model else None
        trainer = Trainer(
            amortizer=amortizer,
            generative_model=self.sim_model.generative_model,
            configurator=configurator,
            learning_rate=inference_settings[self.sim_model_name]["training"]["lr"],
            checkpoint_path=save_model_path,
        )
        return trainer

    def get_evaluater(self):
        sim_model = self.get_sim_model()
        amortizer = self.get_amortizer()
        trainer = self.get_trainer()
        evaluater = Evaluater(
            sim_model,
            amortizer,
            trainer,
            simulation_settings[self.sim_model_name]["plot_settings"],
            inference_settings[self.sim_model_name]["evaluation"],
        )
        return evaluater

    def generate_hdf5_data(self) -> None:
        print("Start generating data:")
        parent_folder = self.file_manager("data")
        pathlib.Path(parent_folder).mkdir(parents=True, exist_ok=True)
        save_path = os.path.join(parent_folder, "data.h5")

        chunk_size = inference_settings[self.sim_model_name]["generate_data"][
            "chunk_size"
        ]
        total_n_sim = inference_settings[self.sim_model_name]["generate_data"][
            "total_n_sim"
        ]
        num_chunks = total_n_sim // chunk_size

        prior_sum = 0
        prior_sqsum = 0
        chunk_sum = 0
        chunk_sqsum = 0

        with h5py.File(save_path, "w") as hf:
            hf.attrs["sim_model_name"] = self.sim_model_name
            hf.attrs["total_n_sim"] = total_n_sim
            hf.attrs["chunk_size"] = chunk_size
            hf.attrs["num_chunks"] = num_chunks
            hf.attrs["is_pde"] = self.sim_model.is_pde
            hf.attrs["sim_data_shape"] = self.sim_model.get_sim_data_shape()

            true_params, sim_data = self.sim_model.generative_model(
                batch_size=chunk_size
            )

            prior_sum += np.mean(true_params)
            prior_sqsum += np.mean(np.square(true_params))
            chunk_sum += np.mean(sim_data)
            chunk_sqsum += np.mean(np.square(sim_data))

            hf.create_dataset(
                "true_params",
                data=true_params,
                compression="gzip",
                chunks=True,
                maxshape=(None, self.sim_model.num_hidden_params),
            )
            hf.create_dataset(
                "sim_data",
                data=sim_data,
                compression="gzip",
                chunks=True,
                maxshape=tuple([None] + list(self.sim_model.get_sim_data_shape())),
            )

            if num_chunks > 1:
                for i in tqdm(range(num_chunks - 1)):
                    true_params, sim_data = self.sim_model.generative_model(
                        batch_size=chunk_size
                    )

                    prior_sum += np.mean(true_params)
                    prior_sqsum += np.mean(np.square(true_params))
                    chunk_sum += np.mean(sim_data)
                    chunk_sqsum += np.mean(np.square(sim_data))

                    hf["true_params"].resize(
                        (hf["true_params"].shape[0] + true_params.shape[0]), axis=0
                    )
                    hf["true_params"][-true_params.shape[0] :] = true_params

                    hf["sim_data"].resize(
                        (hf["sim_data"].shape[0] + sim_data.shape[0]), axis=0
                    )
                    hf["sim_data"][-sim_data.shape[0] :] = sim_data

            prior_mean = prior_sum / num_chunks
            prior_std = (prior_sqsum / num_chunks - prior_mean**2) ** 0.5
            sim_mean = chunk_sum / num_chunks
            sim_std = (chunk_sqsum / num_chunks - sim_mean**2) ** 0.5

            hf.attrs["prior_mean"] = prior_mean
            hf.attrs["prior_std"] = prior_std
            hf.attrs["sim_mean"] = sim_mean
            hf.attrs["sim_std"] = sim_std
