import os
import sys

import json
import pandas as pd

from allennlp.commands import main

dataset =  sys.argv[1]
model = sys.argv[2]

os.environ['model'] = model
turns = ['single', 'multi', 'concat']
experiments = ['remembert/baselines'] * 2 + ['remembert']

metrics = []
for exp, t in zip(experiments, turns):
    os.environ['turns'] = t
    sys.argv = [
        "allennlp",
        "train",
        f"experiments/{exp}/{dataset}.jsonnet",
        "-f",
        "-s",
        f"saved/{exp}/{model}",
        "--include-package",
        "jdnlp"
    ]
    print()
    main()
    res = json.load(open(f"saved/{exp}{mod}/{model}/metrics.json"))
    metrics.append(res)
    
    
df = pd.DataFrame(metrics)
df.to_csv(f"{exp}-{model}.csv".replace('/', '-'), index=False)
    
    
    