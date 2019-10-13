# python -m jdnlp.run transform-data -t tests/fixtures/data_transforms/data_split.json --include-package jdnlp

import os
import sys

mod_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, mod_path)

from jdnlp.commands import main

exp = sys.argv[1]
config = sys.argv[2]

sys.argv = [
    "jdnlp.run",  # command name, not used by main
    "transform-data",
    "-t", f'datasets/{exp}/{config}.json',
    "--include-package", "jdnlp",
]

main()