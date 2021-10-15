import json
from omegaconf import OmegaConf


def load_config_from_json(filepath):
    # Import config
    with open(filepath, "rb") as f:
        data = json.load(f)

    dot_list = []
    for key in data.keys():
        dot_list.append(f"{key}={data[key]['value']}")

    return OmegaConf.from_dotlist(dot_list)
