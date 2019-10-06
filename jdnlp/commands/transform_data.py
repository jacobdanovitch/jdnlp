import argparse

from jdnlp.data import transform_and_split, TRANSFORMATION_REGISTER
from allennlp.commands import Subcommand

class TransformData(Subcommand):
    def add_subparser(
        self, name: str, parser: argparse._SubParsersAction
    ) -> argparse.ArgumentParser:

        description = """Apply the specified transformations to the specified dataset."""
        subparser = parser.add_parser(name, description=description, help="Transform a dataset.")

        subparser.add_argument(
            "-d", 
            "--dataset_path",
            type=str, 
            required=True,
            help="path to dataset"
        )

        subparser.add_argument(
            "-t",
            "--transform_config_path", 
            help="path to transformation config file"
        )

        subparser.set_defaults(func=transform_and_split)
        return subparser
