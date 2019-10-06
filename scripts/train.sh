export CFG="$(cat experiments/$2.json)"
echo "$CFG"

allennlp train "jdnlp/model_configs/$1.json" -f -s "saved/$2/$1" --overrides "$CFG" --include-package jdnlp