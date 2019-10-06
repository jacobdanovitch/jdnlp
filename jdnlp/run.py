import os
import hydra

import jdnlp


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

from jdnlp.commands import main


def run():
    main(prog="jdnlp")


if __name__ == "__main__":
    run()