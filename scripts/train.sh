#!/bin/bash

#if test -f "experiments/$2.jsonnet"; then
    #export CFG="$(cat experiments/$2.jsonnet)"
#else
    #export CFG="$(cat experiments/$2.json)"
#fi
#
#if test -f "jdnlp/model_configs/$1.jsonnet"; then
    #export MODEL_PATH="jdnlp/model_configs/$1.jsonnet"
#else
    #export MODEL_PATH="jdnlp/model_configs/$1.json"
#fi

export MODEL_PATH="experiments/$1.jsonnet"
allennlp train "$MODEL_PATH" -f --include-package jdnlp # -s "saved/$1"

# echo "python -m allennlp.run train \"jdnlp/model_configs/$1.json\" -f -s \"saved/$2/$1\" --overrides $P_CFG --include-package jdnlp"
# tensorboard --logdir saved/counterspeech/attention/remembert/log --bind_all
# --overrides "$CFG"
