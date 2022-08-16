from bayesflow.default_settings import MetaDictSetting

LINAER_ODE_SYSETEM_LSTM_ARCHITECTURE = MetaDictSetting(
    meta_dict={
        "lstm_units": [64, 64, 64],
        "fc_units": [64],
        "fc_activation": "relu",
        "summary_dim": 64,
    }
)

LINEAR_ODE_SYSTEM_INN_ARCHITECUTURE = {
    "n_coupling_layers": 8,
}
