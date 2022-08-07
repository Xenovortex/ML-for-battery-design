import random
import string

import pytest
from docopt import DocoptExit

import ML_for_Battery_Design.src.main as main

valid_modes = ["train", "generate_data", "plot", "evaluate"]

invalid_input = [
    "",
    "".join(random.choices(string.ascii_letters, k=random.randrange(10))),
]


@pytest.mark.parametrize("valid_mode_input", valid_modes)
def test_main_mode_valid(valid_mode_input):
    args = main.main(valid_mode_input)
    assert sum(args.values()) == 1  # only one mode True, all other mode False
    assert args[valid_mode_input] == 1  # correct mode is True


@pytest.mark.parametrize("invalid_mode_input", invalid_input)
def test_main_mode_invalid(invalid_mode_input):
    with pytest.raises(DocoptExit):
        main.main(invalid_mode_input)


def test_main_mode_multiple_inputs():
    with pytest.raises(DocoptExit):
        main.main(random.choices(valid_modes, k=random.randint(2, len(valid_modes))))
