import sys
import pathfinder
import utils
from configuration import config, set_configuration
from utils_plots import plot_learning_curves

if len(sys.argv) < 2:
    sys.exit("Usage: plot.py <configuration_name>")

config_name = sys.argv[1]
set_configuration(config_name)

# metadata
metadata_dir = utils.get_dir_path('models', pathfinder.METADATA_PATH)
metadata_path = utils.find_model_metadata(metadata_dir, config_name)

metadata = utils.load_pkl(metadata_path)
expid = metadata['experiment_id']

analysis_dir = utils.get_dir_path('analysis', pathfinder.METADATA_PATH)

plot_learning_curves(metadata['losses_eval_train'], metadata['losses_eval_valid'], expid, analysis_dir)

