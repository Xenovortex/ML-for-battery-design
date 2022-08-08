import random
import string

from ML_for_Battery_Design.src.simulation.simulation_model import SimulationModel

dummy_param_names = [
    "".join(random.choices(string.ascii_letters, k=random.randrange(10)))
    for i in range(random.randrange(10))
]

dummy_hidden_params = {}
for name in dummy_param_names:
    dummy_hidden_params["sample_" + name] = random.choice([True, False])

dummy_simulation_settings = {
    "dt0": random.uniform(0, 10),
    "max_time_iter": random.randrange(1000),
    "nr": random.randrange(1000),
}

dummy_sample_boundaries = {}
for name in dummy_param_names:
    dummy_sample_boundaries[name] = tuple(
        sorted([random.uniform(-1000, 1000), random.uniform(-1000, 1000)])
    )

dummy_default_values = {}
for name in dummy_param_names:
    dummy_default_values[name] = random.uniform(-1000, 1000)


def test_simulation_model_init_random_input():
    test_object = SimulationModel(
        dummy_hidden_params,
        dummy_simulation_settings,
        dummy_sample_boundaries,
        dummy_default_values,
    )
    assert test_object.hidden_params == dummy_hidden_params
    assert test_object.simulation_settings == dummy_simulation_settings
    assert test_object.sample_boundaries == dummy_sample_boundaries
    assert test_object.default_param_values == dummy_default_values
