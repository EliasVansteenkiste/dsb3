"""
Run with:
python submit.py [-p mypredictions] [-c myconfigfile]
"""
import argparse
from application.submission import generate_submission
from utils.configuration import set_configuration
import utils

if __name__ == "__main__":
    NotImplementedError()
    parser = argparse.ArgumentParser(description=__doc__)
    required = parser.add_argument_group('required arguments')
    required.add_argument('-c', '--config',
                          help='configuration to run',
                          required=True)
    optional = parser.add_argument_group('optional arguments')
    optional.add_argument('-m', '--metadata',
                          help='metadatafile to use',
                          required=False)

    args = parser.parse_args()
    set_configuration(args.config)

    expid = utils.generate_expid(args.config)

    generate_submission(expid)