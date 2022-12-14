from bayesflow.default_settings import MetaDictSetting
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

# ---------------------------------------------------------------------------- #
#                              Simulation Settings                             #
# ---------------------------------------------------------------------------- #


HIDDEN_PARAMS = {
    "sample_a": True,
    "sample_b": True,
    "sample_c": True,
    "sample_d": True,
    "sample_u0": True,
    "sample_v0": True,
}

SIMULATION_SETTINGS = {
    "dt0": 0.1,
    "max_time_iter": 32,
    "use_reject_sampling": True,
    "reject_bound_real": [(0, float("inf"))],
    "reject_bound_complex": None,
    "use_complex_part": True,
}

SAMPLE_BOUNDARIES = {
    "a": (-10, 10),
    "b": (-10, 10),
    "c": (-10, 10),
    "d": (-10, 10),
    "u0": (-10, 10),
    "v0": (-10, 10),
}

DEFAULT_VALUES = {"a": 1, "b": 0, "c": 0, "d": 1, "u0": 1, "v0": 1}

PLOT_SETTINGS = {
    "num_plots": 8,
    "figsize": (15, 10),
    "font_size": 12,
    "show_title": True,
    "show_plot": False,
    "show_time": None,
    "show_params": True,
    "show_eigen": False,
}


LINEAR_ODE_SYSTEM_SIMULATION_SETTINGS = {
    "hidden_params": HIDDEN_PARAMS,
    "simulation_settings": SIMULATION_SETTINGS,
    "sample_boundaries": SAMPLE_BOUNDARIES,
    "default_param_values": DEFAULT_VALUES,
    "plot_settings": PLOT_SETTINGS,
}


# ---------------------------------------------------------------------------- #
#                             Architecture Settings                            #
# ---------------------------------------------------------------------------- #

LINEAR_ODE_SYSTEM_FC_ARCHITECTURE = MetaDictSetting(
    meta_dict={"units": [32, 32, 32], "activation": "relu", "summary_dim": 32}
)

LINEAR_ODE_SYSTEM_LSTM_ARCHITECTURE = MetaDictSetting(
    meta_dict={
        "lstm_units": [32, 32, 32],
        "fc_units": [32, 32],
        "fc_activation": "relu",
        "summary_dim": 32,
    }
)

LINEAR_ODE_SYSTEM_INN_ARCHITECTURE = {
    "n_coupling_layers": 8,
}

LINEAR_ODE_SYSTEM_ARCHITECTURES = {
    "FC": LINEAR_ODE_SYSTEM_FC_ARCHITECTURE,
    "LSTM": LINEAR_ODE_SYSTEM_LSTM_ARCHITECTURE,
    "INN": LINEAR_ODE_SYSTEM_INN_ARCHITECTURE,
}


# ---------------------------------------------------------------------------- #
#                               Training Settings                              #
# ---------------------------------------------------------------------------- #


LINEAR_ODE_SYSTEM_TRAINING_SETTINGS = {
    "lr": PiecewiseConstantDecay(
        [50000, 100000, 150000],
        [0.001, 0.0001, 0.00001, 0.000001],
    ),
    "num_epochs": 200,
    "it_per_epoch": 1000,
    "batch_size": 32,
    "no_bayesflow": False,
}

LINEAR_ODE_SYSTEM_PROCESSING_SETTINGS = {
    "norm_prior": True,
    "norm_sim_data": "mean_std",
    "remove_nan": True,
    "float32_cast": True,
}


LINEAR_ODE_SYSTEM_HDF5_SETTINGS = {"total_n_sim": 320000, "chunk_size": 1000}

LINEAR_ODE_SYSTEM_EVALUATION_SETTINGS = {
    "batch_size": 300,
    "n_samples": 1000,
    "plot_prior": True,
    "plot_sim_data": True,
    "plot_loss": True,
    "plot_latent": True,
    "plot_sbc_histogram": True,  # wait for bayesflow bug resolve
    "plot_sbc_ecdf": True,
    "plot_true_vs_estimated": True,
    "plot_posterior": True,
    "plot_post_with_prior": True,
    "plot_resimulation": True,
}

LINEAR_ODE_SYSTEM_INFERENCE_SETTINGS = {
    "processing": LINEAR_ODE_SYSTEM_PROCESSING_SETTINGS,
    "generate_data": LINEAR_ODE_SYSTEM_HDF5_SETTINGS,
    "training": LINEAR_ODE_SYSTEM_TRAINING_SETTINGS,
    "evaluation": LINEAR_ODE_SYSTEM_EVALUATION_SETTINGS,
}
