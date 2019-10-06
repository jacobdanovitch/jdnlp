export CUDA_DEVICE=0
export DATA_READER="twtc_reader"

export TRAIN_PATH="datasets/twtc/train.csv"
export TEST_PATH="datasets/twtc/test.csv"

allennlp train "experiments/$1.json" -s "saved/$1" -f --include-package jdnlp