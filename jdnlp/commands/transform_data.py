import argparse
import pandas as pd

from allennlp.commands import Subcommand
from allennlp.common.params import Params

from jdnlp.data import DataTransform #, data_transform_from_args
from jdnlp.utils.cli import from_params_with_check
from jdnlp.utils.data import get_pd_fn_from_path

import logging
logger = logging.getLogger(__name__)

class TransformData(Subcommand):
    def add_subparser(
        self, name: str, parser: argparse._SubParsersAction
    ) -> argparse.ArgumentParser:

        description = """Apply the specified transformations to the specified dataset."""
        subparser = parser.add_parser(name, description=description, help="Transform a dataset.")

        subparser.add_argument(
            "-t",
            "--transform_config_path",
            type=str,
            required=True,
            help="path to transformation config file"
        )

        subparser.add_argument(
            "-o", 
            "--overrides",
            type=str,
            required=False,
            help="optional overrides"
        )

        subparser.set_defaults(func=data_transform_from_args)
        return subparser


def data_transform_from_args(args):
    return data_transform_from_file(
        transform_config_path=args.transform_config_path,
        overrides=args.overrides
    )



def data_transform_from_file(
    transform_config_path: str,
    overrides: str = "",
) -> DataTransform:
    params = Params.from_file(transform_config_path, overrides)
    return transform_data(params)


def transform_data(params: Params):
    write_config = params.pop('write_config', ())
    tf = DataTransform.from_params(params)
    dfs = tf.transform()
    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]

    for df, cfg in zip(dfs, write_config):
        fp = cfg.pop('path')
        write_fn = get_pd_fn_from_path(fp, 'to', module=pd.DataFrame)
        write_fn(df, fp, **dict(cfg))

    return dfs
