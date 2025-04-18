import yaml


def read_config(file_path: str = "config.yaml") -> dict:
    """
    Reads a YAML configuration file and returns its content as a dictionary.

    Args:
        file_path (str): The path to the YAML configuration file.

    Returns:
        dict: The content of the YAML file as a dictionary.
    """
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


if __name__ == "__main__":
    config = read_config()
    print("Configuration loaded:")
    print(config)
