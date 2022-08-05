# ML_for_Battery_Design/src/main.py

"""Provide command-line user interface.

Usage:
    main.py train
    main.py generate_data
    main.py plot
    main.py evaluate
"""

from typing import Optional, Sequence

from docopt import docopt


def main(argv: Optional[Sequence[str]] = None) -> dict:
    args = docopt(__doc__, argv)

    print("User input:", args)

    if bool(args["train"]):
        pass
    elif bool(args["generate_data"]):
        pass
    elif bool(args["plot"]):
        pass
    elif bool(args["evaluate"]):
        pass
    else:
        raise ValueError("main.py: Invalid user input!")

    return args


def test():
    return True


if __name__ == "__main__":
    main()
