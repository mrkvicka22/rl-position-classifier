import os
import sys

script_path = os.path.dirname(os.path.realpath(__file__))
root_path = script_path + "/../"
sys.path.insert(0, root_path)

from net import LAYERS_ONES, LAYERS_THREES, LAYERS_TWOS, Net
ONES_MODEL = Net(LAYERS_ONES)
TWOS_MODEL = Net(LAYERS_TWOS)
THREES_MODEL = Net(LAYERS_THREES)
