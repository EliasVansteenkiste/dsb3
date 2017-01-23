"""Module acting as a singleton object for storing the configuration module.
"""

import importlib
import os

_CONFIG_DIR = "configurations"
DEFAULT = "default"
_config = None

_configuration_name = None

def set_configuration(configuration):
    """Imports and initialises the configuration module."""
    global _config, _configuration_name
    if configuration.endswith('.py'):
        configuration = configuration[:-3]
    if os.sep in configuration:
        configuration = os.path.relpath(configuration)
        configuration = configuration.replace(os.sep, '.')
        _configuration_name = configuration
        _config = importlib.import_module(_configuration_name)
    else:
        _configuration_name = "%s.%s" % (_CONFIG_DIR, configuration)
        _config = importlib.import_module(_configuration_name)
    print _configuration_name
    if configuration != DEFAULT:
        print "loaded", _config

set_configuration(DEFAULT)

def get_configuration_name():
    global _configuration_name
    return _configuration_name

# dirty hack to access config attributes as if they were actually in the module
class Configuration():
    def __getattr__(self, name):
        return _config.__getattribute__(name)

config = Configuration()