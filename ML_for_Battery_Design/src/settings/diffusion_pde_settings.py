from bayesflow.default_settings import MetaDictSetting
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

# ---------------------------------------------------------------------------- #
#                              Simulation Settings                             #
# ---------------------------------------------------------------------------- #


HIDDEN_PARAMS = {
    "sample_a": False,
    "sample_b": False,
    "sample_c": False,
    "sample_d": False,
    "sample_Du": False,
    "sample_Dv": False,
    "sample_alpha_u": True,
    "sample_beta_u": True,
    "sample_gamma_u": True,
    "sample_alpha_v": True,
    "sample_beta_v": True,
    "sample_gamma_v": True,
    "sample_u0": False,
    "sample_v0": False,
}

SIMULATION_SETTINGS = {
    "dt0": 0.25,  # 5,
    "max_time_iter": 20,  # 20,  # 100,
    "nr": 10,
    "use_reject_sampling": False,
    "use_f_terms": True,
    "random_initial": False,
    "rtol": None,
    "atol": None,
    "hmin": 0,
    "reduce_2D": True,
}

SAMPLE_BOUNDARIES = {
    "a": (-1, 1),
    "b": (-1, 1),
    "c": (-1, 1),
    "d": (-1, 1),
    "Du": (0, 1),  # (0.005, 0.01),
    "Dv": (0, 1),
    "alpha_u": (0, 1),
    "beta_u": (-1, 0),
    "gamma_u": (-1, 1),
    "alpha_v": (0, 1),
    "beta_v": (-1, 0),
    "gamma_v": (-1, 1),
    "u0": (-10, 10),
    "v0": (-10, 10),
}

DEFAULT_VALUES = {
    "a": 0,
    "b": 0,
    "c": 0,
    "d": 0,
    "Du": 0,
    "Dv": 0,
    "alpha_u": 0,
    "beta_u": 0,
    "gamma_u": 1,
    "alpha_v": 0,
    "beta_v": 0,
    "gamma_v": 1,
    "u0": 0,
    "v0": 0,
}

PLOT_SETTINGS = {
    "num_plots": 8,
    "figsize": (15, 10),
    "font_size": 12,
    "show_title": True,
    "show_plot": False,
    "show_time": None,
}

DIFFUSION_PDE_MODEL_SIMULATION_SETTINGS = {
    "hidden_params": HIDDEN_PARAMS,
    "simulation_settings": SIMULATION_SETTINGS,
    "sample_boundaries": SAMPLE_BOUNDARIES,
    "default_param_values": DEFAULT_VALUES,
    "plot_settings": PLOT_SETTINGS,
}


# ---------------------------------------------------------------------------- #
#                             Architecture Settings                            #
# ---------------------------------------------------------------------------- #

DIFFUSION_PDE_MODEL_FC_ARCHITECTURE = MetaDictSetting(
    meta_dict={"units": [128, 128, 128], "activation": "relu", "summary_dim": 64}
)

DIFFUSION_PDE_MODEL_CNN_ARCHITECTURE = MetaDictSetting(
    meta_dict={
        "num_filters": [32, 64, 128],
        "cnn_activation": "elu",
        "units": [1024, 1024],
        "fc_activation": "relu",
        "summary_dim": 64,
        "pool_time": True,
        "pool_space": True,
    }
)

DIFFUSION_PDE_MODEL_LSTM_ARCHITECTURE = MetaDictSetting(
    meta_dict={
        "lstm_units": [512, 512, 512],
        "fc_units": [512, 512],
        "fc_activation": "relu",
        "summary_dim": 64,
    }
)

DIFFUSION_PDE_MODEL_DOUBLE_LSTM_ARCHITECTURE = MetaDictSetting(
    meta_dict={
        "LSTM": DIFFUSION_PDE_MODEL_LSTM_ARCHITECTURE,
        "FC": DIFFUSION_PDE_MODEL_FC_ARCHITECTURE,
    }
)

DIFFUSION_PDE_MODEL_CONVLSTM_ARCHITECTURE = MetaDictSetting(
    meta_dict={
        "num_filters": [32, 64, 128],
        "units": [1024, 1024],
        "summary_dim": 64,
        "fc_activation": "relu",
        "pool_time": False,
        "pool_space": False,
        "batch_norm": True,
    }
)

DIFFUSION_PDE_MODEL_INN_ARCHITECTURE = {
    "n_coupling_layers": 10,
}

DIFFUSION_PDE_MODEL_ARCHITECTURES = {
    "FC": DIFFUSION_PDE_MODEL_FC_ARCHITECTURE,
    "CNN": DIFFUSION_PDE_MODEL_CNN_ARCHITECTURE,
    "INN": DIFFUSION_PDE_MODEL_INN_ARCHITECTURE,
    "ConvLSTM": DIFFUSION_PDE_MODEL_CONVLSTM_ARCHITECTURE,
    "LSTM": DIFFUSION_PDE_MODEL_LSTM_ARCHITECTURE,
    "DoubleLSTM": DIFFUSION_PDE_MODEL_DOUBLE_LSTM_ARCHITECTURE,
}

# ---------------------------------------------------------------------------- #
#                               Training Settings                              #
# ---------------------------------------------------------------------------- #


DIFFUSION_PDE_MODEL_TRAINING_SETTINGS = {
    "lr": PiecewiseConstantDecay(
        [10000, 20000, 30000],
        [0.001, 0.0001, 0.00001, 0.000001],
    ),
    "num_epochs": 40,
    "it_per_epoch": 1000,
    "batch_size": 32,
    "no_bayesflow": False,
}

DIFFUSION_PDE_MODEL_PROCESSING_SETTINGS = {
    "norm_prior": True,
    "norm_sim_data": "mean_std",
    "remove_nan": True,
    "float32_cast": True,
}

DIFFUSION_PDE_MODEL_HDF5_SETTINGS = {"total_n_sim": 32200000, "chunk_size": 1000}

DIFFUSION_PDE_MODEL_EVALUATION_SETTINGS = {
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

DIFFUSION_PDE_MODEL_INFERENCE_SETTINGS = {
    "processing": DIFFUSION_PDE_MODEL_PROCESSING_SETTINGS,
    "generate_data": DIFFUSION_PDE_MODEL_HDF5_SETTINGS,
    "training": DIFFUSION_PDE_MODEL_TRAINING_SETTINGS,
    "evaluation": DIFFUSION_PDE_MODEL_EVALUATION_SETTINGS,
}
