import os
import json

import argparse
# import hydra

import jdnlp

from allennlp.commands import Subcommand
from allennlp.commands.train import train_model_from_file
from allennlp.common.util import import_submodules

_REQUIRED_PARAMS = ['model', 'exp']


class Train(Subcommand):
    def add_subparser(
        self, name: str, parser: argparse._SubParsersAction
    ) -> argparse.ArgumentParser:

        description = """Train the specified model on the specified dataset."""
        subparser = parser.add_parser(name, description=description, help="Train a model.")

        subparser.add_argument(
            "model", type=str, help="model type/name"
        )

        subparser.add_argument(
            "exp", type=str, help="experiment name"
        )

        subparser.set_defaults(func=train)
        return subparser



def train(cfg):
    # import_submodules("jdnlp")
    
    exp_fp = f'experiments/{cfg.exp}.json'
    if not os.path.isfile(exp_fp):
        raise FileNotFoundError(f'Experiment file {exp_fp} not found in dir {os.getcwd()}')

    params = dict(
        parameter_filename=f'jdnlp/model_configs/{cfg.model}.json',
        serialization_dir=f'saved/{cfg.exp}/{cfg.model}',
        overrides=open(exp_fp).read(),
        force=True,
        cache_directory="~/.cache/allennlp",
    )
    #print(json.dumps(params, indent=2))

    train_model_from_file(**params)





"""
@hydra.main()
def main(cfg):
    # hack to subvert hydra stuff
    module_path = os.path.dirname(jdnlp.__file__)
    module_folder = os.path.dirname(module_path)
    
    os.chdir(module_folder)

    print(cfg.pretty())
    for p in _REQUIRED_PARAMS:
        if not hasattr(cfg, p):
            raise ConfigurationError(f"Missing required parameter {p}.")

    train(cfg)


if __name__ == "__main__":
    main()
"""