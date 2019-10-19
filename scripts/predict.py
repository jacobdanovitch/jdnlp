import os
import sys

import pandas as pd
import numpy as np

from allennlp.commands import main
from sklearn.metrics import classification_report, f1_score

exp = sys.argv[1]
model = sys.argv[2]

input_file = f"datasets/{exp}/json/predict.json"
tmp_file = f'/tmp/{exp}_{model.replace("/", "_")}.json'
output_file = f"saved/{exp}/{model}/test_predictions.json"

truth = pd.read_json(input_file, lines=True)
truth.to_json(tmp_file, orient='records', lines=True)

sys.argv = [
    "allennlp",
    "predict",
    f"saved/{exp}/{model}/model.tar.gz",
    tmp_file,
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

truth = truth.label.values
preds = pd.read_json(output_file, lines=True)# .label.values
preds = preds.class_probabilities.apply(lambda x: int(np.argmax(x)))

print(f'F1: {f1_score(truth, preds, average="macro"):0.4f}')
print(classification_report(truth, preds))

os.remove(tmp_file)