import math
import random
import string

import numpy as np
import pytest

from ML_for_Battery_Design.src.simulation.simulation_model import SimulationModel

dummy_param_names = [
    "".join(random.choices(string.ascii_letters, k=random.randrange(10)))
    for i in range(random.randrange(10))
]

dummy_hidden_params = {}
for name in dummy_param_names:
    dummy_hidden_params["sample_" + name] = random.choice([True, False])

dummy_ode_simulation_settings = {
    "dt0": random.uniform(0, 10),
    "max_time_iter": random.randrange(1000),
}

dummy_pde_simulation_settings = {
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

non_dict_input = [
    random.randint(-1000, 1000),  # int
    random.uniform(-1000, 1000),  # float
    random.uniform(-1000, 1000) + random.uniform(-1000, 1000) * 1j,  # complex
    random.choice([True, False]),  # bool
    tuple(random.random() for x in range(random.randrange(10))),  # tuple
    list(random.random() for x in range(random.randrange(10))),  # list
    set(random.random() for x in range(random.randrange(10))),  # set
    frozenset(random.random() for x in range(random.randrange(10))),  # frozenset
    "".join(random.choices(string.ascii_letters, k=random.randrange(10))),  # string,
    None,  # NoneType
]


def get_concrete_class(AbstractClass, *args):
    class ConcreteClass(AbstractClass):
        def __init__(self, *args) -> None:
            super().__init__(*args)

    ConcreteClass.__abstractmethods__ = frozenset()
    return type("DummyConcreteClassOf" + AbstractClass.__name__, (ConcreteClass,), {})


@pytest.mark.parametrize(
    "simulation_settings",
    [dummy_ode_simulation_settings, dummy_pde_simulation_settings],
)
def test_simulation_model_init_random_valid_input(simulation_settings):
    test_object = get_concrete_class(SimulationModel)(
        dummy_hidden_params,
        simulation_settings,
        dummy_sample_boundaries,
        dummy_default_values,
    )
    assert test_object.hidden_params == dummy_hidden_params
    assert test_object.simulation_settings == simulation_settings
    assert test_object.sample_boundaries == dummy_sample_boundaries
    assert test_object.default_param_values == dummy_default_values


def test_simulation_model_init_ode():
    test_object = get_concrete_class(SimulationModel)(
        dummy_hidden_params,
        dummy_ode_simulation_settings,
        dummy_sample_boundaries,
        dummy_default_values,
    )
    assert test_object.dt0 == dummy_ode_simulation_settings["dt0"]
    assert test_object.max_time_iter == dummy_ode_simulation_settings["max_time_iter"]
    assert not test_object.is_pde


def test_simulation_model_init_pde():
    test_object = get_concrete_class(SimulationModel)(
        dummy_hidden_params,
        dummy_pde_simulation_settings,
        dummy_sample_boundaries,
        dummy_default_values,
    )
    assert test_object.dt0 == dummy_pde_simulation_settings["dt0"]
    assert test_object.max_time_iter == dummy_pde_simulation_settings["max_time_iter"]
    assert test_object.nr == dummy_pde_simulation_settings["nr"]
    assert test_object.is_pde


@pytest.mark.parametrize("non_dict_input", non_dict_input)
@pytest.mark.parametrize(
    "simulation_settings",
    [dummy_ode_simulation_settings, dummy_pde_simulation_settings],
)
def test_simulation_model_non_dict_hidden_params(non_dict_input, simulation_settings):
    with pytest.raises(TypeError):
        get_concrete_class(SimulationModel)(
            non_dict_input,
            simulation_settings,
            dummy_sample_boundaries,
            dummy_default_values,
        )
    with pytest.raises(TypeError):
        get_concrete_class(SimulationModel)(
            dummy_hidden_params,
            non_dict_input,
            dummy_sample_boundaries,
            dummy_default_values,
        )
    with pytest.raises(TypeError):
        get_concrete_class(SimulationModel)(
            dummy_hidden_params,
            simulation_settings,
            non_dict_input,
            dummy_default_values,
        )
    with pytest.raises(TypeError):
        get_concrete_class(SimulationModel)(
            dummy_hidden_params,
            simulation_settings,
            dummy_sample_boundaries,
            non_dict_input,
        )


@pytest.mark.parametrize(
    "simulation_settings",
    [dummy_ode_simulation_settings, dummy_pde_simulation_settings],
)
def test_simulation_model_abstract_methods(simulation_settings):
    test_object = get_concrete_class(SimulationModel)(
        dummy_hidden_params,
        simulation_settings,
        dummy_sample_boundaries,
        dummy_default_values,
    )
    with pytest.raises(NotImplementedError):
        test_object.get_sim_data_dim()
    with pytest.raises(NotImplementedError):
        test_object.simulator()
    with pytest.raises(NotImplementedError):
        test_object.plot_sim_data()


@pytest.mark.parametrize(
    "simulation_settings",
    [dummy_ode_simulation_settings, dummy_pde_simulation_settings],
)
def test_simulation_model_get_time_points_method(simulation_settings):
    test_object = get_concrete_class(SimulationModel)(
        dummy_hidden_params,
        simulation_settings,
        dummy_sample_boundaries,
        dummy_default_values,
    )
    time_points = test_object.get_time_points()
    print()
    assert len(time_points.shape) == 1
    assert time_points.shape[0] == simulation_settings["max_time_iter"]
    assert np.any(np.diff(time_points) == np.diff(time_points)[0])
    assert math.isclose(np.diff(time_points)[0], simulation_settings["dt0"])


@pytest.mark.parametrize(
    "simulation_settings",
    [dummy_ode_simulation_settings, dummy_pde_simulation_settings],
)
def test_simulation_model_get_param_names_method(simulation_settings):
    test_object = get_concrete_class(SimulationModel)(
        dummy_hidden_params,
        simulation_settings,
        dummy_sample_boundaries,
        dummy_default_values,
    )
    hidden_param_names = test_object.get_param_names()
    assert len(hidden_param_names) == sum(dummy_hidden_params.values())
    for key, value in dummy_hidden_params.items():
        if value:
            assert key[len("sample_") :] in hidden_param_names
        else:
            assert key[len("sample_") :] not in hidden_param_names


@pytest.mark.parametrize(
    "simulation_settings",
    [dummy_ode_simulation_settings, dummy_pde_simulation_settings],
)
def test_simulation_model_get_default_param_kwargs_method(simulation_settings):
    test_object = get_concrete_class(SimulationModel)(
        dummy_hidden_params,
        simulation_settings,
        dummy_sample_boundaries,
        dummy_default_values,
    )
    default_param_kwargs = test_object.get_default_param_kwargs()
    assert len(default_param_kwargs) == (
        len(dummy_hidden_params) - sum(dummy_hidden_params.values())
    )
    for key, value in dummy_hidden_params.items():
        if not value:
            assert key[len("sample_") :] in default_param_kwargs
            assert (
                default_param_kwargs[key[len("sample_") :]]
                == dummy_default_values[key[len("sample_") :]]
            )
        else:
            assert key[len("sample_") :] not in default_param_kwargs
