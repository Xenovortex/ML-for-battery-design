# ML_for_Battery_Design/src/main.py

"""Provide command-line user interface.

Usage:
    main.py train_online <sim_model> <summary_net> [<filename>] [-t | --test_mode] [-k | --skip_wrappers]
    main.py train_online <sim_model> <summary_net> <filename> [-s | --save_model] [-t | --test_mode] [-k | --skip_wrappers]
    main.py train_offline <sim_model> <data_name> <summary_net> [<filename>] [-t | --test_mode] [-k | --skip_wrappers]
    main.py train_offline <sim_model> <data_name> <summary_net> <filename> [-s | --save_model] [-t | --test_mode] [-k | --skip_wrappers]
    main.py generate_data <sim_model> <data_name> [-t | --test_mode] [-k | --skip_wrappers]
    main.py analyze_sim <sim_model> [<data_name> <filename>] [-t | --test_mode] [-k | --skip_wrappers]
    main.py evaluate <sim_model> <data_name> <filename> [-s | --save_model] [-t | --test_mode] [-k | --skip_wrappers]
    main.py -h | --help

Options:
    -h, --help              Show this screen.
    -s, --save_model        Save trained BayesFlow model.
    -t, --test_mode         Reduce runtime for unit testing
    -k, --skip_wrappers     Skip wrappers execution for unit testing
"""

from typing import Optional, Sequence

from docopt import docopt
from tabulate import tabulate

from ML_for_Battery_Design.src.helpers.wrappers import (
    analyze_sim,
    evaluate,
    generate_data,
    train_offline,
    train_online,
)


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
        train_online(**args)
    elif bool(args["train_offline"]):
        train_offline(**args)
    elif bool(args["generate_data"]):
        generate_data(**args)
    elif bool(args["analyze_sim"]):
        analyze_sim(**args)
    elif bool(args["evaluate"]):
        evaluate(**args)

    return args


if __name__ == "__main__":
    main()  # pragma: no cover
