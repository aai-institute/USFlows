import typing as T
import os
import uuid
import pandas as pd
from datetime import datetime


class Experiment(object):
    """Base class for experiments. 
    """
    def __init__(self, name, *args, **kwargs):
        super().__init__()
        self.name = name
    
    @classmethod 
    def _init_rec(cls, cfg):
        if isinstance(cfg, dict):
            if "experiment" in cfg:
                experiment_type = cfg["experiment"]["experiment_type"]
                params = cls._init_rec(cfg["experiment"]["experiment_params"])
                
                return experiment_type(**params)
            else:
                return {k: cls._init_rec(v) for k, v in cfg.items()}
        elif isinstance(cfg, list):
            return [cls._init_rec(v) for v in cfg]
        else:
            return cfg
    
    @classmethod
    def from_dict(cls, config: T.Dict[str, T.Any]) -> "Experiment":
        if "experiment" not in config:
            raise ValueError("Invalid config file. The config file needs to contain an experiment field.")
        return cls._init_rec(config)
    
    def conduct(self, report_dir: os.PathLike, storage_path: os.PathLike = None) -> None:
        """Conducts the experiment and saves the results to the report directory. The method is expected to store all results in report_dir.
        """
        raise NotImplementedError
    