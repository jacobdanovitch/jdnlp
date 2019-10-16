# python -m jdnlp.run transform-data -t tests/fixtures/data_transforms/data_split.json --include-package jdnlp

import os
import sys

from tqdm.auto import tqdm

mod_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, mod_path)

from jdnlp.commands import main

exp = sys.argv[1]
transforms = list(sys.argv[2:])

pbar = tqdm(transforms)
for t in pbar:
    pbar.set_description(t)
    sys.argv = [
        "jdnlp.run",  # command name, not used by main
        "transform-data",
        "-t", f'datasets/{exp}/transforms/{t}.json',
        "--include-package", "jdnlp",
    ]

    main()