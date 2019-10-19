#!/bin/bash
if test -f "experiments/$2.jsonnet"; then
    export CFG="$(cat experiments/$2.jsonnet)"
else
    export CFG="$(cat experiments/$2.json)"
fi

if test -f "jdnlp/model_configs/$1.jsonnet"; then
    export MODEL_PATH="jdnlp/model_configs/$1.jsonnet"
else
    export MODEL_PATH="jdnlp/model_configs/$1.json"
fi


# echo "python -m allennlp.run train \"jdnlp/model_configs/$1.json\" -f -s \"saved/$2/$1\" --overrides $P_CFG --include-package jdnlp"

allennlp train "$MODEL_PATH" -f -s "saved/$2/$1" --overrides "$CFG" --include-package jdnlp
