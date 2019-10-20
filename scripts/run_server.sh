python -m allennlp.service.server_simple \
            --archive-path https://storage.googleapis.com/jacobdanovitch/twtc/lstm.tar.gz \
            --predictor text_classifier \
            --include-package jdnlp \
            --field-name sentence \
            --title "Trouble with the Curve"

"""
from allennlp.models.archival import load_archive 
from allennlp.service.predictors import Predictor
from allennlp.common.util import import_submodules

import_submodules(\"jdnlp\")
archive = load_archive(\"saved/model.tar.gz\")
pred = Predictor.from_archive(archive, 'text_classifier')
"""