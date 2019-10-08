#!/bin/bash

json_escape () {
    printf '%s' "$1" | python -c 'import json,sys; print(json.dumps(sys.stdin.read()))'
}

export CFG="$(cat experiments/$2.json)"
# echo "$CFG"

export P_CFG=`printf '%s' "$CFG" | python -c 'import json,sys; print(json.dumps(sys.stdin.read()))'`
echo "python -m allennlp.run train \"jdnlp/model_configs/$1.json\" -f -s \"saved/$2/$1\" --overrides $P_CFG --include-package jdnlp"

allennlp train "jdnlp/model_configs/$1.json" -f -s "saved/$2/$1" --overrides $CFG --include-package jdnlp
