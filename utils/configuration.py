"""Module acting as a singleton object for storing the configuration module.
"""

import importlib
import os

DEFAULT = "configurations.default"
_config = None

_configuration_name = None


def path_to_importable_string(path):
    if path.endswith('.py'):
        path = path[:-3]
        if os.sep in path:
            path = os.path.relpath(path)
            path = path.replace(os.sep, '.')
    else:
        pass
    return path

def set_configuration(configuration):
    """Imports and initialises the configuration module."""
    global _config, _configuration_name
    configuration = path_to_importable_string(configuration)
    _configuration_name = configuration
    _config = importlib.import_module(_configuration_name)
    if configuration != DEFAULT:
        print "loaded", _config

# set_configuration(DEFAULT)

def get_configuration_name():
    global _configuration_name
    return _configuration_name

# dirty hack to access config attributes as if they were actually in the module
class Configuration():
    def __getattr__(self, name):
        return _config.__getattribute__(name)

config = Configuration()