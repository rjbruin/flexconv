from omegaconf import OmegaConf
import pandas as pd


def flatten_configdict(
    cfg: OmegaConf,
    sep: str = ".",
):
    cfgdict = OmegaConf.to_container(cfg)
    cfgdict = pd.json_normalize(cfgdict, sep=sep)

    return cfgdict.to_dict(orient="records")[0]
