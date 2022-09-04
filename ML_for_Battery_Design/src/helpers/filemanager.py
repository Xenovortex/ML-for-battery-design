import os


class FileManager:
    """Managing file saving path structure based on file type (data, model, result)

    Attributes:
        mode (str): main file execution mode (Options: train_online, train_offline, generate_data, analyze_sim, evaluate)
        sim_model_name (str): name of simulation model
        summary_net_name (str): name of summary network
        data_name (str): name of dataset
        filename (str): filename provided by user from main user interface
    """

    def __init__(self, mode: str, **kwargs: str) -> None:
        """Initializes :class:FileManager

        Args:
            mode (str): main file execution mode (Options: train_online, train_offline, generate_data, analyze_sim, evaluate)
            **kwargs (dict): keyword arguments from docopt user interface
        """
        self.mode = mode
        self.sim_model_name = kwargs["<sim_model>"]
        self.summary_net_name = kwargs["<summary_net>"]
        self.data_name = kwargs["<data_name>"]
        self.filename = kwargs["<filename>"]

    def __call__(self, file_type: str) -> str:
        """Generate save file path based on file type (data, model, result)

        Args:
            file_type (str): type of file to generate a save path for. Options: data, model, result

        Returns:
            path (str): save path for given file type
        """
        if file_type == "data":
            path = os.path.join("data", self.sim_model_name, self.data_name)
        elif file_type == "model":
            path = os.path.join(
                "models", self.sim_model_name, self.data_name, self.filename
            )
        elif file_type == "result":
            if self.data_name is not None:
                path = os.path.join(
                    "results", self.sim_model_name, self.data_name, self.filename
                )
            else:
                path = os.path.join("results", self.sim_model_name)
        else:
            raise ValueError(
                "{} - call: {} is not a valid file_type. Valid options are: data, model, result".format(
                    self.__class__.__name__, file_type
                )
            )
        return path


"""

    def generate_config_summary(self, file_type: str, path: str) -> None:
        if file_type == "data":
            with open(os.path.join(path, "config_info.txt"), "w") as file:
                pass
        elif file_type == "model":
            with open(os.path.join(path, "config_info.txt"), "w") as file:
                pass
        elif file_type == "result":
            with open(os.path.join(path, "config_info.txt"), "w") as file:
                pass
        else:
            raise ValueError(
                "{} - generate_config_summary: {} is not a valid file_type. Valid options are: data, model, result".format(
                    self.__class__.__name__, file_type
                )
            )

"""
