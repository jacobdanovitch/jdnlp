import json
import shutil
import sys

from allennlp.commands import main

config_file = "jdnlp/model_configs/attention/HAN.json"

# Use overrides to train on CPU.
# overrides = open("tests/fixtures/twtc_classifier.json").read()
# print(json.dumps(overrides))
overrides = open("experiments/twtc.json").read()

serialization_dir = "/tmp/debugger_train"

# Training will fail if the serialization directory already
# has stuff in it. If you are running the same training loop
# over and over again for debugging purposes, it will.
# Hence we wipe it out in advance.
# BE VERY CAREFUL NOT TO DO THIS FOR ACTUAL TRAINING!
shutil.rmtree(serialization_dir, ignore_errors=True)

# Assemble the command into sys.argv
sys.argv = [
    "allennlp",  # command name, not used by main
    "train",
    config_file,
    "-s", serialization_dir,
    "--include-package", "jdnlp",
    "-o", overrides,
]

main()