import argparse
from overrides import overrides

from allennlp.common.params import Params
from allennlp.common.checks import ConfigurationError

class ArgumentParserWithDefaults(argparse.ArgumentParser):
    """
    Custom argument parser that will display the default value for an argument
    in the help message.
    """

    _action_defaults_to_ignore = {"help", "store_true", "store_false", "store_const"}

    @staticmethod
    def _is_empty_default(default):
        if default is None:
            return True
        if isinstance(default, (str, list, tuple, set)):
            return not bool(default)
        return False

    @overrides
    def add_argument(self, *args, **kwargs):

        # Add default value to the help message when the default is meaningful.
        default = kwargs.get("default")
        if kwargs.get(
            "action"
        ) not in self._action_defaults_to_ignore and not self._is_empty_default(default):
            description = kwargs.get("help") or ""
            kwargs["help"] = f"{description} (default = {default})"
        super().add_argument(*args, **kwargs)



def from_params_with_check(params: Params, param_name: str, message_override: str = None):
    param = params.get(param_name, None)
    if not param:
        raise ConfigurationError(f'Must specify {message_override or param_name}.')
    
    return param