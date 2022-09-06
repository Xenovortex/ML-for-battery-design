from bayesflow.default_settings import MetaDictSetting
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

# ---------------------------------------------------------------------------- #
#                              Simulation Settings                             #
# ---------------------------------------------------------------------------- #

HIDDEN_PARAMS = {
    "sample_C_rate": False,
    "sample_L": False,
    "sample_eps": False,
    "sample_r": False,
    "sample_Ds": True,
    "sample_k": False,
}

SIMULATION_SETTINGS = {
    "dt0": 375.0,
    "max_time_iter": 20,
    "nr": 20,
    "stop_condition": 1,
    "V_cut": 3,
    "use_reject_sampling": False,
}

SAMPLE_BOUNDARIES = {
    "C_rate": (0.1, 2),
    "L": (20e-6, 200e-6),
    "eps": (0.4, 0.7),
    "r": (1e-6, 10e-6),
    "Ds": (5e-14, 1e-13),
    "k": (10e-12, 10e-10),
}

DEFAULT_VALUES = {
    "C_rate": 0.5,
    "L": 65.5e-6,
    "eps": 0.6287064,
    "r": 5.0e-6,
    "Ds": 1e-14,
    "k": 1.044546283497307e-11,
}

PLOT_SETTINGS = {
    "num_plots": 8,
    "figsize": (15, 10),
    "font_size": 12,
    "show_title": True,
    "show_plot": False,
    "show_time": None,
    "show_params": True,
    "show_eigen": True,
}

SPM_BATTERY_MODEL_SIMULATION_SETTINGS = {
    "hidden_params": HIDDEN_PARAMS,
    "simulation_settings": SIMULATION_SETTINGS,
    "sample_boundaries": SAMPLE_BOUNDARIES,
    "default_param_values": DEFAULT_VALUES,
    "plot_settings": PLOT_SETTINGS,
}


# ---------------------------------------------------------------------------- #
#                             Architecture Settings                            #
# ---------------------------------------------------------------------------- #

SPM_BATTERY_MODEL_FC_ARCHITECTURE = MetaDictSetting(
    meta_dict={"units": [32, 32, 32], "activation": "relu", "summary_dim": 32}
)

SPM_BATTERY_MODEL_CNN_ARCHITECTURE = MetaDictSetting(
    meta_dict={
        "num_filters": [32, 64, 128],
        "cnn_activation": "elu",
        "units": [1024, 1024],
        "fc_activation": "relu",
        "summary_dim": 128,
        "pool_time": True,
        "pool_space": True,
    }
)

SPM_BATTERY_MODEL_CONVLSTM_ARCHITECTURE = MetaDictSetting(
    meta_dict={
        "num_filters": [32, 64, 128],
        "units": [1024, 1024],
        "summary_dim": 128,
        "fc_activation": "relu",
        "pool_time": True,
        "pool_space": True,
        "batch_norm": True,
    }
)

SPM_BATTERY_MODEL_INN_ARCHITECTURE = {
    "n_coupling_layers": 8,
}

SPM_BATTERY_MODEL_ARCHITECTURES = {
    "FC": SPM_BATTERY_MODEL_FC_ARCHITECTURE,
    "CNN": SPM_BATTERY_MODEL_CNN_ARCHITECTURE,
    "INN": SPM_BATTERY_MODEL_INN_ARCHITECTURE,
}


# ---------------------------------------------------------------------------- #
#                               Training Settings                              #
# ---------------------------------------------------------------------------- #

SPM_BATTERY_MODEL_TRAINING_SETTINGS = {
    "lr": PiecewiseConstantDecay([100000, 150000], [0.001, 0.0001, 0.00001]),
    "num_epochs": 200,
    "it_per_epoch": 1000,
    "batch_size": 32,
}

SPM_BATTERY_MODEL_PROCESSING_SETTINGS = {
    "norm_prior": True,
    "norm_sim_data": "log_norm",
    "remove_nan": True,
    "float32_cast": True,
}

SPM_BATTERY_MODEL_HDF5_SETTINGS = {"total_n_sim": 1024000, "chunk_size": 10240}

SPM_BATTERY_MODEL_EVALUATION_SETTINGS = {
    "batch_size": 300,
    "n_samples": 100,
    "plot_prior": True,
    "plot_sim_data": True,
    "plot_loss": True,
    "plot_latent": True,
    "plot_sbc_histogram": False,  # wait for bayesflow bug resolve
    "plot_sbc_ecdf": True,
    "plot_true_vs_estimated": True,
    "plot_posterior": True,
    "plot_post_with_prior": True,
    "plot_resimulation": True,
}

SPM_BATTERY_MODEL_INFERENCE_SETTINGS = {
    "processing": SPM_BATTERY_MODEL_PROCESSING_SETTINGS,
    "generate_data": SPM_BATTERY_MODEL_HDF5_SETTINGS,
    "training": SPM_BATTERY_MODEL_TRAINING_SETTINGS,
    "evaluation": SPM_BATTERY_MODEL_EVALUATION_SETTINGS,
}
