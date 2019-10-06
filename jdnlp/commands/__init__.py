"""
Lots pulled from https://github.com/allenai/allennlp/blob/master/allennlp/commands/__init__.py.
"""

from typing import *
import argparse

from allennlp.commands import Subcommand

from jdnlp.commands.train import Train
from jdnlp.utils.cli import ArgumentParserWithDefaults

def create_parser(
    prog: str = None, subcommand_overrides: Dict[str, Subcommand] = None
) -> argparse.ArgumentParser:
    """
    Creates the argument parser for the main program.
    """
    if subcommand_overrides is None:
        subcommand_overrides = {}

    parser = ArgumentParserWithDefaults(description="Run jdnlp", usage="%(prog)s", prog=prog)
    # parser.add_argument("--version", action="version", version="%(prog)s " + __version__)

    subparsers = parser.add_subparsers(title="Commands", metavar="")

    subcommands = {
        # Default commands
        # "transform-data": TransformData()
        "train": Train(),
        # "evaluate": Evaluate(),
        **subcommand_overrides,
    }

    for name, subcommand in subcommands.items():
        subparser = subcommand.add_subparser(name, subparsers)

    return parser


def main(prog: str = None, subcommand_overrides: Dict[str, Subcommand] = None) -> None:
    if subcommand_overrides is None:
        subcommand_overrides = {}

    parser = create_parser(prog, subcommand_overrides)
    args = parser.parse_args()

    # If a subparser is triggered, it adds its work as `args.func`.
    # So if no such attribute has been added, no subparser was triggered,
    # so give the user some help.
    if "func" in dir(args):
        # Import any additional modules needed (to register custom classes).
        # for package_name in getattr(args, "include_package", ()):
        #    import_submodules(package_name)
        args.func(args)
    else:
        parser.print_help()