from typing import Iterable
import pandas as pd
import os
import click
import json

def build_report(expdir, report_file, config_prefix=""):
    report = None
    for d in os.listdir(expdir):
        if os.path.isdir(expdir + "/" + d):
            try:
                with open(expdir + "/" + d + "/result.json", "r") as f:
                    result = json.loads("{\"test_" + f.read().split("{\"test_")[-1])
            except:
                continue
            
            config = result["config"]
            for k in config.keys():
                result[config_prefix + k] = config[k] if not isinstance(config[k], Iterable) else str(config[k])
            result.pop("config")

            if report is None:
                report = pd.DataFrame(result, index=[0])
            else:
                report = pd.concat([report, pd.DataFrame(result, index=[0])], ignore_index=True)
    
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    report.to_csv(report_file, index=False)

@click.command()
@click.option("--expdir", help="Experiment directory")
@click.option("--report_file", help="Report file")
@click.option("--config_prefix", default="", help="Prefix for config items")
def _build_report(expdir, report_file, config_prefix):
    build_report(expdir, report_file, config_prefix)

if __name__ == "__main__":
    _build_report()
    