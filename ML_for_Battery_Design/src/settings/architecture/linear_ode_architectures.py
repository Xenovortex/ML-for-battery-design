from bayesflow.default_settings import MetaDictSetting

LINAER_ODE_SYSETEM_LSTM_ARCHITECTURE = MetaDictSetting(
    meta_dict={
        "lstm_units": [64, 64, 64],
        "fc_units": [64],
        "fc_activation": "relu",
        "summary_dim": 64,
    }
)

LINEAR_ODE_SYSTEM_INN_ARCHITECTURE = {
    "n_coupling_layers": 8,
}

LINEAR_ODE_SYSETEM_ARCHITECTURES = {
    "LSTM": LINAER_ODE_SYSETEM_LSTM_ARCHITECTURE,
    "INN": LINEAR_ODE_SYSTEM_INN_ARCHITECTURE,
}
