import os


class FileManager:
    def __init__(self, mode: str, **kwargs: str) -> None:
        self.mode = mode
        self.sim_model_name = kwargs["<sim_model>"]
        self.summary_net_name = kwargs["<summary_net>"]
        self.data_name = kwargs["<data_name>"]
        self.filename = kwargs["<filename>"]

    def __call__(self, file_type: str) -> str:
        if file_type == "data":
            path = os.path.join("data", self.sim_model_name, self.data_name)
        elif file_type == "model":
            path = os.path.join(
                "model", self.sim_model_name, self.data_name, self.filename
            )
        elif file_type == "result":
            path = os.path.join(
                "results", self.sim_model_name, self.data_name, self.filename
            )
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
