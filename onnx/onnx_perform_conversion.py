import os
import sys
import torch

script_path = os.path.dirname(os.path.realpath(__file__))
root_path = script_path + "/../"
sys.path.insert(0, root_path)
sys.path.insert(0, script_path)

from onnx_inference_models import TWOS_MODEL
from settings import TWOS_MODEL_PATH, TWOS_MODEL_PATH_ONNX

def convert_twos():
  pytorch_training_model = torch.load(TWOS_MODEL_PATH)
  state_dict = pytorch_training_model.state_dict()

  pytorch_inference_model = TWOS_MODEL
  pytorch_inference_model.load_state_dict(state_dict)
  pytorch_inference_model.eval()
  dummy_input = torch.zeros(12)
  torch.onnx.export(pytorch_inference_model, dummy_input, TWOS_MODEL_PATH_ONNX, verbose=True)

if __name__ == '__main__':
  convert_twos()