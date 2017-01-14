"""Module for reading the SETTINGS.json file.
"""

import json
import os

with open(os.path.dirname(os.path.realpath(__file__)) + '/../SETTINGS.json') as data_file:
    PATHS = json.load(data_file)
    globals().update(PATHS)