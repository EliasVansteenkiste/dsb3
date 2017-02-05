import importlib

_config = None


def set_configuration(config_dir, config_name):
    global _config
    _config = importlib.import_module("%s.%s" % (config_dir, config_name))
    print "Loaded", _config


def config():
    return _config
