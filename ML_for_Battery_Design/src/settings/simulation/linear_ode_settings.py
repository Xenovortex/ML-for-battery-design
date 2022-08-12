HIDDEN_PARAMS = {
    "sample_a": True,
    "sample_b": True,
    "sample_c": True,
    "sample_d": True,
    "sample_u0": True,
    "sample_v0": True,
}

SIMULATION_SETTINGS = {"dt0": 0.1, "max_time_iter": 30, "use_reject_sampling": True}

SAMPLE_BOUNDARIES = {
    "a": (-10, 10),
    "b": (-10, 10),
    "c": (-10, 10),
    "d": (-10, 10),
    "u0": (1, 10),
    "v0": (1, 10),
}

DEFAULT_VALUES = {"a": 1, "b": 0, "c": 0, "d": 1, "u0": 1, "v0": 1}

PLOT_SETTINGS = {
    "num_plots": 8,
    "figsize": (15, 10),
    "font_size": 12,
    "show_title": True,
    "show_plot": False,
    "show_params": True,
    "show_eigen": True,
}

LINEAR_ODE_SYSTEM_SETTINGS = {
    "hidden_params": HIDDEN_PARAMS,
    "simulation_settings": SIMULATION_SETTINGS,
    "sample_boundaries": SAMPLE_BOUNDARIES,
    "default_param_values": DEFAULT_VALUES,
    "plot_settings": PLOT_SETTINGS,
}
