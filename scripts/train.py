# https://github.com/allenai/allennlp/issues/3391

import os, sys
import subprocess
import argparse

import json
import pandas as pd

from allennlp.commands import main

import torchsnooper
import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def get_args():
    # https://www.mattzeunert.com/2012/02/12/unnamed-arguments-with-argparse.html
    parser = argparse.ArgumentParser()
    parser.add_argument('exp', nargs=1, type=str)
    parser.add_argument('-s', '--suffix', required=False, type=str)

    # https://stackoverflow.com/questions/37367331/is-it-possible-to-use-argparse-to-capture-an-arbitrary-set-of-optional-arguments
    parsed, unknown = parser.parse_known_args()
    environ = []
    for arg in unknown:
        if arg.startswith(("-", "--")):
            arg = arg.split('=')[0]
            parser.add_argument(arg)
            environ.append(arg.replace('-', ''))
    
    args = parser.parse_args()
    args.exp = args.exp[0]

    return args, environ


if __name__ == '__main__':
    args, environ = get_args()
    config_vars = []
    for arg, val in vars(args).items():
        if arg in environ:
            os.environ[arg] = val
            # config_vars.append(f'{arg}={val}')
            config_vars.append((arg, val))

    config_vars = '_'.join(sorted((val for (_, val) in config_vars), key=lambda x: x[0]))
    folder_name = config_vars + (f'_{args.suffix}' if args.suffix else '')
    sys.argv = [
        "allennlp",
        "train",
        f"experiments/{args.exp}.jsonnet",
        "-f",
        "-s",
        f"saved/{args.exp}/{folder_name}",
        "--include-package",
        "jdnlp"
    ]

    logger.warning(' '.join(sys.argv))
    proc = subprocess.run(sys.argv,
        stdin = sys.stdin, #subprocess.PIPE,
        stdout=sys.stdout, #subprocess.PIPE,
        stderr=sys.stderr, #subprocess.PIPE,
        universal_newlines=True,
        bufsize=0)
    
    # main()