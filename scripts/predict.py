import os
import sys

import pandas as pd
import numpy as np

from allennlp.commands import main
from sklearn.metrics import classification_report

exp = sys.argv[1]
model = sys.argv[2]

input_file = f"datasets/{exp}/json/test.json"
output_file = f"saved/{exp}/{model}/test_predictions.json"

sys.argv = [
    "allennlp",
    "predict",
    f"saved/{exp}/{model}/model.tar.gz",
    input_file,
    "--include-package",
    "jdnlp",
    "--cuda-device",
    "0",
    "--predictor",
    "text_classifier",
    "--silent",
    "--output-file",
    output_file
]

main()

truth = pd.read_json(input_file, lines=True).label.values
preds = pd.read_json(output_file, lines=True)# .label.values
preds = preds.class_probabilities.apply(lambda x: int(np.argmax(x)))

print(classification_report(truth, preds))
