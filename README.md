# jdnlp
A collection of NLP models, modules and dataset readers built with AllenNLP with a focus on rapid experimentation and benchmarking. 

Some implementations are partially or primarily conversions of others' work to AllenNLP; they have been credited in-line where applicable. 

## Models

| Model                              | Registerable     |
|------------------------------------|------------------|
| Bag-Of-Embeddings                  | `pooling/boe`    |
| TextCNN                            | `cnn/text_cnn`   |
| LSTM + Self-Attention              | `lstm/attn_lstm` |
| Biattentive Classification Network | `attention/bcn`  |
| Hierarchical Attention Network     | `attention/han`  |

## Projects

### [Trouble with the Curve](https://github.com/jacobdanovitch/Trouble-With-The-Curve)

To replicate the experiments from this work, run:

```shell
sh scripts/train.sh <Registerable> twtc
```
