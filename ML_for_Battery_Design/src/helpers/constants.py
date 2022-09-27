from ML_for_Battery_Design.src.helpers.summary import (
    CNN_Network,
    ConvLSTM_Network,
    DoubleLSTM_Network,
    FC_Network,
    LSTM_Network,
    SPM_Network,
)
from ML_for_Battery_Design.src.settings.diffusion_pde_settings import (
    DIFFUSION_PDE_MODEL_ARCHITECTURES,
    DIFFUSION_PDE_MODEL_INFERENCE_SETTINGS,
    DIFFUSION_PDE_MODEL_SIMULATION_SETTINGS,
)
from ML_for_Battery_Design.src.settings.linear_ode_settings import (
    LINEAR_ODE_SYSTEM_ARCHITECTURES,
    LINEAR_ODE_SYSTEM_INFERENCE_SETTINGS,
    LINEAR_ODE_SYSTEM_SIMULATION_SETTINGS,
)
from ML_for_Battery_Design.src.settings.SPM_battery_settings import (
    SPM_BATTERY_MODEL_ARCHITECTURES,
    SPM_BATTERY_MODEL_INFERENCE_SETTINGS,
    SPM_BATTERY_MODEL_SIMULATION_SETTINGS,
)
from ML_for_Battery_Design.src.simulation.diffusion_pde_model import DiffusionPDEModel
from ML_for_Battery_Design.src.simulation.linear_ode_model import LinearODEsystem
from ML_for_Battery_Design.src.simulation.SPM_battery import SPMBatteryModel

summary_collection = {
    "FC": FC_Network,
    "LSTM": LSTM_Network,
    "CNN": CNN_Network,
    "ConvLSTM": ConvLSTM_Network,
    "SPM": SPM_Network,
    "DoubleLSTM": DoubleLSTM_Network,
}

sim_model_collection = {
    "linear_ode_system": LinearODEsystem,
    "SPM": SPMBatteryModel,
    "diffusion_pde": DiffusionPDEModel,
}

architecture_settings = {
    "linear_ode_system": LINEAR_ODE_SYSTEM_ARCHITECTURES,
    "SPM": SPM_BATTERY_MODEL_ARCHITECTURES,
    "diffusion_pde": DIFFUSION_PDE_MODEL_ARCHITECTURES,
    "pytest": {"pytest": {}},  # needed for unit testing
}

simulation_settings = {
    "linear_ode_system": LINEAR_ODE_SYSTEM_SIMULATION_SETTINGS,
    "SPM": SPM_BATTERY_MODEL_SIMULATION_SETTINGS,
    "diffusion_pde": DIFFUSION_PDE_MODEL_SIMULATION_SETTINGS,
    "pytest": {},  # needed for unit testing
}

inference_settings = {
    "linear_ode_system": LINEAR_ODE_SYSTEM_INFERENCE_SETTINGS,
    "SPM": SPM_BATTERY_MODEL_INFERENCE_SETTINGS,
    "diffusion_pde": DIFFUSION_PDE_MODEL_INFERENCE_SETTINGS,
    "pytest": {},  # needed for unit testing
}
