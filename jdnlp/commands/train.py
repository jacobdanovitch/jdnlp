import os
import json
import hydra

import jdnlp

from allennlp.commands.train import train_model_from_file

_REQUIRED_PARAMS = ['model', 'exp']

def train(cfg):
    exp_fp = f'experiments/{cfg.exp}.json'
    if not os.path.isfile(exp_fp):
        raise FileNotFoundError(f'Experiment file {exp_fp} not found in dir {os.getcwd()}')

    params = dict(
        parameter_filename=f'jdnlp/model_configs/{cfg.model}.json',
        serialization_dir=f'saved/{cfg.exp}/{cfg.model}',
        overrides=open(exp_fp).read(),
        force=True,
        cache_directory="~/.cache/allennlp"
    )
    print(json.dumps(params, indent=2))

    train_model_from_file(**params)



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