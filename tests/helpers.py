import ML_for_Battery_Design.src.main as main


def get_concrete_class(AbstractClass, *args):
    """Create concrete child of abstract class for unit testing"""

    class ConcreteClass(AbstractClass):
        def __init__(self, *args) -> None:
            super().__init__(*args)

    ConcreteClass.__abstractmethods__ = frozenset()
    return type("DummyConcreteClassOf" + AbstractClass.__name__, (ConcreteClass,), {})


def setup_user_args(mode, sim_model, save_model=False):
    if mode == "train_online":
        user_input = [mode, sim_model, "FC", "pytest_file"]
        if save_model:
            user_input += ["--save_model"]
    elif mode == "train_offline":
        user_input = [mode, sim_model, "pytest_data", "FC", "pytest_file"]
        if save_model:
            user_input += ["--save_model"]
    elif mode == "generate_data":
        user_input = [mode, sim_model, "pytest_data"]
    elif mode == "analyze_sim":
        user_input = [mode, sim_model, "pytest_file"]
    elif mode == "evaluate":
        user_input = [mode, sim_model, "pytest_data", "pytest_file"]
    else:
        raise ValueError("{} is not a valid mode".format(mode))

    user_input += ["--test_mode"]

    args = main.main(user_input)

    return args
