from ML_for_Battery_Design.src.helpers.summary import (
    CNN_Network,
    FC_Network,
    LSTM_Network,
)
from ML_for_Battery_Design.src.settings.linear_ode_settings import (
    LINEAR_ODE_SYSTEM_ARCHITECTURES,
    LINEAR_ODE_SYSTEM_INFERENCE_SETTINGS,
    LINEAR_ODE_SYSTEM_SIMULATION_SETTINGS,
)
from ML_for_Battery_Design.src.simulation.linear_ode_model import LinearODEsystem

summary_collection = {
    "FC": FC_Network,
    "LSTM": LSTM_Network,
    "CNN": CNN_Network,
}

sim_model_collection = {"linear_ode_system": LinearODEsystem}

architecture_settings = {
    "linear_ode_system": LINEAR_ODE_SYSTEM_ARCHITECTURES,
    "pytest": {"pytest": {}},  # needed for unit testing
}

simulation_settings = {
    "linear_ode_system": LINEAR_ODE_SYSTEM_SIMULATION_SETTINGS,
    "pytest": {},  # needed for unit testing
}

inference_settings = {
    "linear_ode_system": LINEAR_ODE_SYSTEM_INFERENCE_SETTINGS,
    "pytest": {},  # needed for unit testing
}
