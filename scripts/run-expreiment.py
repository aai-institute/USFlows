import click
import os 
import typing as T
from laplace_flows.experiments.utils import read_config

Pathable = T.Union[str, os.PathLike] # In principle one can cast it to os.path.Path

@click.command()
@click.option("--report_dir", default="./", help="Report file")
@click.option("--config", default="./config.yaml", help="Prefix for config items")
def run(report_dir: Pathable, config: Pathable):
    """Loads an experiment from config file conducts the experiment it.
    
    Args:
        report_dir (str): Directory to save report to.
        config (str): Path to config file. The report is expected to be specified in .yaml format with
        support to some special key functionalities (see :func:`~laplace_flows.experiments.utils.read_config)
        Defaults to "./config.yaml".
    """
    sepline = "\n" + ("-" * 80) + "\n"
    print(
        f"{sepline}Parsing config file:"
        )
    config = os.path.abspath(config)
    experiment = read_config(config)
    print(
        f"Done.{sepline}"
        )
    print(
        f"{sepline}Conducting experiment"
        )
    experiment.conduct(report_dir)
    print(
        f"Done.{sepline}"
        )
    
if __name__ == "__main__":
    run()