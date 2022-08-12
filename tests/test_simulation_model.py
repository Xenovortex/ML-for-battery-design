import random
import string

import numpy as np
import pytest
from bayesflow.forward_inference import GenerativeModel, Prior, Simulator

from ML_for_Battery_Design.src.simulation.simulation_model import SimulationModel
from tests.helpers import get_concrete_class

# ------------------------------ Dummy Test Data ----------------------------- #


dummy_param_names = [
    "".join(random.choices(string.ascii_letters, k=random.randrange(10)))
    for i in range(0, random.randint(2, 10))
]

dummy_hidden_params = {}
for name in dummy_param_names:
    dummy_hidden_params["sample_" + name] = random.choice([True, False])
true_param_idx = list(dummy_hidden_params.keys())[
    random.randrange(len(dummy_hidden_params))
]
if sum(dummy_hidden_params.values()) == 0:
    dummy_hidden_params[true_param_idx] = True

dummy_no_hidden_params = {}
for name in dummy_param_names:
    dummy_no_hidden_params["sample_" + name] = False

dummy_hidden_params_true_false = dummy_hidden_params
while True:
    false_param_idx = list(dummy_hidden_params.keys())[
        random.randrange(len(dummy_hidden_params))
    ]
    if false_param_idx != true_param_idx:
        break
if sum(dummy_hidden_params_true_false.values()) == 0:
    dummy_hidden_params_true_false[false_param_idx] = False

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

# ----------------- Test SimulationModel Class Initialization ---------------- #


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
    non_dict_input, simulation_settings, capsys
):
    with pytest.raises(TypeError):
        test_object = get_concrete_class(SimulationModel)(
            non_dict_input,
            simulation_settings,
            dummy_sample_boundaries,
            dummy_default_values,
        )
        out, err = capsys.readouterr()
        assert out == ""
        assert err == "{}: hidden_params input is not dictionary type".format(
            test_object.__class__.__name__
        )
    with pytest.raises(TypeError):
        test_object = get_concrete_class(SimulationModel)(
            dummy_hidden_params,
            non_dict_input,
            dummy_sample_boundaries,
            dummy_default_values,
        )
        out, err = capsys.readouterr()
        assert out == ""
        assert err == "{}: simulation_settings input is not dictionary type".format(
            test_object.__class__.__name__
        )
    with pytest.raises(TypeError):
        test_object = get_concrete_class(SimulationModel)(
            dummy_hidden_params,
            simulation_settings,
            non_dict_input,
            dummy_default_values,
        )
        out, err = capsys.readouterr()
        assert out == ""
        assert err == "{}: sample_boundaries input is not dictionary type".format(
            test_object.__class__.__name__
        )
    with pytest.raises(TypeError):
        test_object = get_concrete_class(SimulationModel)(
            dummy_hidden_params,
            simulation_settings,
            dummy_sample_boundaries,
            non_dict_input,
        )
        out, err = capsys.readouterr()
        assert out == ""
        assert err == "{}: default_param_values input is not dictionary type".format(
            test_object.__class__.__name__
        )


def test_simulation_model_init_ode():
    test_object = get_concrete_class(SimulationModel)(
        dummy_hidden_params,
        dummy_ode_simulation_settings,
        dummy_sample_boundaries,
        dummy_default_values,
    )
    assert isinstance(test_object.dt0, float)
    assert isinstance(test_object.max_time_iter, int)
    assert isinstance(test_object.is_pde, bool)
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
    assert isinstance(test_object.dt0, float)
    assert isinstance(test_object.max_time_iter, int)
    assert isinstance(test_object.nr, int)
    assert isinstance(test_object.is_pde, bool)
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
    assert isinstance(test_object.t, np.ndarray)
    assert isinstance(test_object.hidden_param_names, list)
    assert isinstance(test_object.default_param_kwargs, dict)
    assert np.array_equal(test_object.t, test_object.get_time_points())
    assert test_object.hidden_param_names == test_object.get_param_names()
    assert test_object.default_param_kwargs == test_object.get_default_param_kwargs()


@pytest.mark.parametrize(
    "simulation_settings",
    [dummy_ode_simulation_settings, dummy_pde_simulation_settings],
)
def test_simulation_model_init_bayesflow_prior(simulation_settings):
    test_object = get_concrete_class(SimulationModel)(
        dummy_hidden_params,
        simulation_settings,
        dummy_sample_boundaries,
        dummy_default_values,
    )
    assert isinstance(test_object.prior, Prior)


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


# --------------------------- Test Abstract Methods -------------------------- #


@pytest.mark.parametrize(
    "simulation_settings",
    [dummy_ode_simulation_settings, dummy_pde_simulation_settings],
)
def test_simulation_model_abstract_methods(simulation_settings, capsys):
    test_object = get_concrete_class(SimulationModel)(
        dummy_hidden_params,
        simulation_settings,
        dummy_sample_boundaries,
        dummy_default_values,
    )
    with pytest.raises(NotImplementedError):
        test_object.get_sim_data_dim()
        out, err = capsys.readouterr()
        assert out == ""
        assert err == "{}: get_sim_data_dim method is not implement".format(
            test_object.__class__.__name__
        )
    with pytest.raises(NotImplementedError):
        params = np.random.uniform(-1000, 1000, size=sum(dummy_hidden_params.values()))
        test_object.solver(params)
        out, err = capsys.readouterr()
        assert out == ""
        assert err == "{}: solver method is not implement".format(
            test_object.__class__.__name__
        )
    with pytest.raises(NotImplementedError):
        test_object.plot_sim_data()
        out, err = capsys.readouterr()
        assert out == ""
        assert err == "{}: plot_sim_data is not implement".format(
            test_object.__class__.__name__
        )


# ------------------------------- Test Methods ------------------------------- #


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
    assert np.all(np.isclose(np.diff(time_points), simulation_settings["dt0"]))


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
    "hidden_params",
    [dummy_hidden_params, dummy_hidden_params_true_false],
)
def test_simulation_model_print_internal_settings_method_ode(hidden_params, capsys):
    test_object = get_concrete_class(SimulationModel)(
        hidden_params,
        dummy_ode_simulation_settings,
        dummy_sample_boundaries,
        dummy_default_values,
    )

    def dummy_get_sim_data_dim_ode():
        return (dummy_ode_simulation_settings["max_time_iter"], 2)

    setattr(test_object, "get_sim_data_dim", dummy_get_sim_data_dim_ode)

    expected_output = (
        80 * "#"
        + "\n"
        + "\n"
        + "Initialize simulation model: {}\n".format(type(test_object).__name__)
        + 80 * "-"
        + "\n"
        + "hidden parameters: {}\n".format(test_object.hidden_param_names)
        + "dt0: {}\n".format(test_object.dt0)
        + "max_time_iter: {}\n".format(test_object.max_time_iter)
        + "simulation data dimensions: {}\n".format((test_object.max_time_iter, 2))
        + "\n"
        + "parameter values:\n"
    )

    for key, value in hidden_params.items():
        if value:
            expected_output += "{}: {} -> boundary\n".format(
                key[len("sample_") :], dummy_sample_boundaries[key[len("sample_") :]]
            )
        else:
            expected_output += "{}: {} -> constant\n".format(
                key[len("sample_") :], dummy_default_values[key[len("sample_") :]]
            )

    test_object.print_internal_settings()
    out, err = capsys.readouterr()
    assert out == expected_output
    assert err == ""


@pytest.mark.parametrize(
    "hidden_params",
    [dummy_hidden_params, dummy_hidden_params_true_false],
)
def test_simulation_model_print_internal_settings_method_pde(hidden_params, capsys):
    test_object = get_concrete_class(SimulationModel)(
        hidden_params,
        dummy_pde_simulation_settings,
        dummy_sample_boundaries,
        dummy_default_values,
    )

    def dummy_get_sim_data_dim_pde():
        return (
            dummy_pde_simulation_settings["max_time_iter"],
            dummy_pde_simulation_settings["nr"],
            2,
        )

    setattr(test_object, "get_sim_data_dim", dummy_get_sim_data_dim_pde)

    expected_output = (
        80 * "#"
        + "\n"
        + "\n"
        + "Initialize simulation model: {}\n".format(type(test_object).__name__)
        + 80 * "-"
        + "\n"
        + "hidden parameters: {}\n".format(test_object.hidden_param_names)
        + "dt0: {}\n".format(test_object.dt0)
        + "max_time_iter: {}\n".format(test_object.max_time_iter)
        + "nr: {}\n".format(test_object.nr)
        + "simulation data dimensions: {}\n".format(
            (test_object.max_time_iter, test_object.nr, 2)
        )
        + "\n"
        + "parameter values:\n"
    )

    for key, value in hidden_params.items():
        if value:
            expected_output += "{}: {} -> boundary\n".format(
                key[len("sample_") :], dummy_sample_boundaries[key[len("sample_") :]]
            )
        else:
            expected_output += "{}: {} -> constant\n".format(
                key[len("sample_") :], dummy_default_values[key[len("sample_") :]]
            )

    test_object.print_internal_settings()
    out, err = capsys.readouterr()
    assert out == expected_output
    assert err == ""


@pytest.mark.parametrize(
    "simulation_settings",
    [dummy_ode_simulation_settings, dummy_pde_simulation_settings],
)
def test_simulation_model_sample_to_kwargs_method(simulation_settings):
    test_object = get_concrete_class(SimulationModel)(
        dummy_hidden_params,
        simulation_settings,
        dummy_sample_boundaries,
        dummy_default_values,
    )
    dummy_sample = np.random.uniform(
        -1000, 1000, size=sum(dummy_hidden_params.values())
    ).astype(np.float32)
    param_kwargs = test_object.sample_to_kwargs(dummy_sample)
    assert isinstance(param_kwargs, dict)
    assert len(param_kwargs) == len(dummy_hidden_params)
    assert len(param_kwargs) == len(dummy_sample_boundaries)
    assert len(param_kwargs) == len(dummy_default_values)
    counter = 0
    for key, value in param_kwargs.items():
        if dummy_hidden_params["sample_" + key]:
            assert value == dummy_sample[counter]
            counter += 1
        else:
            assert value == dummy_default_values[key]


@pytest.mark.parametrize(
    "simulation_settings",
    [dummy_ode_simulation_settings, dummy_pde_simulation_settings],
)
def test_simulation_model_reject_sampler(simulation_settings):
    test_object = get_concrete_class(SimulationModel)(
        dummy_hidden_params,
        simulation_settings,
        dummy_sample_boundaries,
        dummy_default_values,
    )
    dummy_sample = np.random.uniform(
        -1000, 1000, size=sum(dummy_hidden_params.values())
    ).astype(np.float32)
    assert not test_object.reject_sampler(dummy_sample)


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


def test_simulation_model_get_bayesflow_amortizer_ode():
    test_object = get_concrete_class(SimulationModel)(
        dummy_hidden_params,
        dummy_ode_simulation_settings,
        dummy_sample_boundaries,
        dummy_default_values,
    )

    def dummy_solver(params):
        return np.random.uniform(
            -1000, 1000, size=(dummy_ode_simulation_settings["max_time_iter"], 2)
        )

    setattr(test_object, "solver", dummy_solver)

    prior, simulator, generative_model = test_object.get_bayesflow_amortizer()
    batch_size = random.randint(1, 8)
    data_dict = generative_model(batch_size=batch_size)
    prior_samples = data_dict["prior_draws"]
    sim_data = data_dict["sim_data"]
    assert isinstance(prior, Prior)
    assert isinstance(simulator, Simulator)
    assert isinstance(generative_model, GenerativeModel)
    assert len(prior_samples.shape) == 2
    assert prior_samples.shape[0] == batch_size
    assert prior_samples.shape[1] == sum(dummy_hidden_params.values())
    assert len(sim_data.shape) == 3
    assert sim_data.shape[0] == batch_size
    assert sim_data.shape[1] == dummy_ode_simulation_settings["max_time_iter"]
    assert sim_data.shape[2] == 2


def test_simulation_model_get_bayesflow_amortizer_pde():
    test_object = get_concrete_class(SimulationModel)(
        dummy_hidden_params,
        dummy_pde_simulation_settings,
        dummy_sample_boundaries,
        dummy_default_values,
    )

    def dummy_solver(params):
        return np.random.uniform(
            -1000,
            1000,
            size=(
                dummy_pde_simulation_settings["max_time_iter"],
                dummy_pde_simulation_settings["nr"],
                2,
            ),
        )

    setattr(test_object, "solver", dummy_solver)

    prior, simulator, generative_model = test_object.get_bayesflow_amortizer()
    batch_size = random.randint(1, 8)
    data_dict = generative_model(batch_size=batch_size)
    prior_samples = data_dict["prior_draws"]
    sim_data = data_dict["sim_data"]
    assert isinstance(prior, Prior)
    assert isinstance(simulator, Simulator)
    assert isinstance(generative_model, GenerativeModel)
    assert len(prior_samples.shape) == 2
    assert prior_samples.shape[0] == batch_size
    assert prior_samples.shape[1] == sum(dummy_hidden_params.values())
    assert len(sim_data.shape) == 4
    assert sim_data.shape[0] == batch_size
    assert sim_data.shape[1] == dummy_pde_simulation_settings["max_time_iter"]
    assert sim_data.shape[2] == dummy_pde_simulation_settings["nr"]
    assert sim_data.shape[3] == 2


@pytest.mark.parametrize(
    "simulation_settings",
    [dummy_ode_simulation_settings, dummy_pde_simulation_settings],
)
def test_simulation_model_get_prior_means_stds(simulation_settings):
    test_object = get_concrete_class(SimulationModel)(
        dummy_hidden_params,
        simulation_settings,
        dummy_sample_boundaries,
        dummy_default_values,
    )

    setattr(
        test_object,
        "prior",
        Prior(
            prior_fun=test_object.uniform_prior,
            param_names=test_object.get_param_names(),
        ),
    )

    prior_means, prior_stds = test_object.get_prior_means_stds()

    assert isinstance(prior_means, np.ndarray)
    assert isinstance(prior_stds, np.ndarray)
    assert len(prior_means.shape) == 2
    assert prior_means.shape[0] == 1
    assert prior_means.shape[1] == sum(dummy_hidden_params.values())
    assert len(prior_stds.shape) == 2
    assert prior_stds.shape[0] == 1
    assert prior_stds.shape[1] == sum(dummy_hidden_params.values())
