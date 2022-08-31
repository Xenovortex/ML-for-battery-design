# ML_for_Battery_Design/src/main.py

"""Provide command-line user interface.

Usage:
    main.py train_online <sim_model> <summary_net> [<filename>] [-t | --test_mode]
    main.py train_online <sim_model> <summary_net> <filename> [-s | --save_model] [-t | --test_mode]
    main.py train_offline <sim_model> <data_name> <summary_net> [<filename>] [-t | --test_mode]
    main.py train_offline <sim_model> <data_name> <summary_net> <filename> [-s | --save_model] [-t | --test_mode]
    main.py generate_data <sim_model> <data_name> [-t | --test_mode]
    main.py analyze_sim <sim_model> [<filename>] [-t | --test_mode]
    main.py evaluate <sim_model> <data_name> <filename> [-t | --test_mode]
    main.py -h | --help

Options:
    -h, --help          Show this screen.
    -s, --save_model    Save trained BayesFlow model.
    -t, --test_mode     Reduce runtime for unit testing
"""

from typing import Optional, Sequence

from docopt import docopt
from tabulate import tabulate

# from ML_for_Battery_Design.src.helpers.wrappers import train_online


def main(argv: Optional[Sequence[str]] = None) -> dict:
    """Main function for running the command line interface

    Args:
        argv (Optional[Sequence[str]], optional): argument vector/list input passed by command line. Defaults to None.

    Returns:
        args (dict): user input given through command line
    """
    args = docopt(__doc__, argv)

    print("Interface user input:")
    print(tabulate(list(args.items()), missingval="None"))

    if bool(args["train_online"]):
        pass  # train_online(**args)
    elif bool(args["train_offline"]):
        pass
    elif bool(args["generate_data"]):
        pass
    elif bool(args["analyze_sim"]):
        pass
    elif bool(args["evaluate"]):
        pass

    return args


if __name__ == "__main__":
    main()  # pragma: no cover
