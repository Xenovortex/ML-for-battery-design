import math
import random
import string

import numpy as np
import pytest

from ML_for_Battery_Design.src.simulation.simulation_model import SimulationModel
from tests.helpers import get_concrete_class

dummy_param_names = [
    "".join(random.choices(string.ascii_letters, k=random.randrange(10)))
    for i in range(random.randrange(10))
]

dummy_hidden_params = {}
for name in dummy_param_names:
    dummy_hidden_params["sample_" + name] = random.choice([True, False])

dummy_no_hidden_params = {}
for name in dummy_param_names:
    dummy_no_hidden_params["sample_" + name] = False

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


@pytest.mark.parametrize("non_dict_input", non_dict_input)
@pytest.mark.parametrize(
    "simulation_settings",
    [dummy_ode_simulation_settings, dummy_pde_simulation_settings],
)
def test_simulation_model_init_non_dict_hidden_params(
    non_dict_input, simulation_settings
):
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


@pytest.mark.parametrize(
    "simulation_settings",
    [dummy_ode_simulation_settings, dummy_pde_simulation_settings],
)
def test_simulation_model_init_method_calls(simulation_settings):
    test_object = get_concrete_class(SimulationModel)(
        dummy_hidden_params,
        simulation_settings,
        dummy_sample_boundaries,
        dummy_default_values,
    )
    assert np.array_equal(test_object.t, test_object.get_time_points())
    assert test_object.hidden_param_names == test_object.get_param_names()
    assert test_object.default_param_kwargs == test_object.get_default_param_kwargs()


@pytest.mark.parametrize(
    "simulation_settings",
    [dummy_ode_simulation_settings, dummy_pde_simulation_settings],
)
def test_simulation_model_init_no_param_warning(simulation_settings, capsys):
    test_object = get_concrete_class(SimulationModel)(
        dummy_no_hidden_params,
        simulation_settings,
        dummy_sample_boundaries,
        dummy_default_values,
    )
    out, err = capsys.readouterr()
    assert out == "Warning: {} - No hidden parameters to sample.\n".format(
        type(test_object).__name__
    )
    assert err == ""


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
    assert isinstance(time_points, np.ndarray)
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
    assert isinstance(hidden_param_names, list)
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
    assert isinstance(default_param_kwargs, dict)
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


@pytest.mark.parametrize(
    "simulation_settings",
    [dummy_ode_simulation_settings, dummy_pde_simulation_settings],
)
def test_simulation_model_sample_to_kwargs_method(simulation_settings):
    dummy_samples = np.array(
        [random.uniform(-1000, 1000) for x in range(sum(dummy_hidden_params.values()))]
    ).astype(np.float32)
    test_object = get_concrete_class(SimulationModel)(
        dummy_hidden_params,
        simulation_settings,
        dummy_sample_boundaries,
        dummy_default_values,
    )
    param_kwargs = test_object.sample_to_kwargs(dummy_samples)
    assert isinstance(param_kwargs, dict)
    assert len(param_kwargs) == len(dummy_hidden_params)
    assert len(param_kwargs) == len(dummy_sample_boundaries)
    assert len(param_kwargs) == len(dummy_default_values)
    counter = 0
    for key, value in dummy_hidden_params.items():
        if value:
            assert param_kwargs[key[len("sample_") :]] == dummy_samples[counter]
            counter += 1
        else:
            assert (
                param_kwargs[key[len("sample_") :]]
                == dummy_default_values[key[len("sample_") :]]
            )


@pytest.mark.parametrize(
    "simulation_settings",
    [dummy_ode_simulation_settings, dummy_pde_simulation_settings],
)
def test_simulation_model_uniform_prior_method(simulation_settings):
    test_object = get_concrete_class(SimulationModel)(
        dummy_hidden_params,
        simulation_settings,
        dummy_sample_boundaries,
        dummy_default_values,
    )

    def dummy_reject_sampler(self, sample):
        return random.choice([True, False])

    setattr(SimulationModel, "reject_sampler", dummy_reject_sampler)

    sample = test_object.uniform_prior(reject_sampling=False)
    sample_reject = test_object.uniform_prior(reject_sampling=True)

    assert isinstance(sample, np.ndarray)
    assert sample.dtype == np.float32
    assert len(sample.shape) == 1
    assert sample.shape[0] == sum(dummy_hidden_params.values())
    counter = 0
    for name, (lower_boundary, upper_boundary) in dummy_sample_boundaries.items():
        if dummy_hidden_params["sample_" + name]:
            assert sample[counter] >= lower_boundary
            assert sample[counter] <= upper_boundary
            counter += 1

    assert isinstance(sample_reject, np.ndarray)
    assert sample_reject.dtype == np.float32
    assert len(sample_reject.shape) == 1
    assert sample_reject.shape[0] == sum(dummy_hidden_params.values())
    counter = 0
    for name, (lower_boundary, upper_boundary) in dummy_sample_boundaries.items():
        if dummy_hidden_params["sample_" + name]:
            assert sample_reject[counter] >= lower_boundary
            assert sample_reject[counter] <= upper_boundary
            counter += 1
