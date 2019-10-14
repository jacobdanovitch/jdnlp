wget -O datasets/twtc/labelled.csv https://github.com/jacobdanovitch/Trouble-With-The-Curve/blob/master/notebooks/preprocessed.csv?raw=true
python scripts/transform_data.py twtc split valid_split resample augment

# python scripts/transform_data.py twtc split
# python scripts/transform_data.py twtc valid_split
# python scripts/transform_data.py twtc augment
# python scripts/transform_data.py twtc resample