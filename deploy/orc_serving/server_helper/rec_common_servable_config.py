import os
import sys
from os.path import dirname
from typing import List

import numpy as np
from mindspore_serving.server import register

from .model_process_helper import ModelProcessor

current_file_path = os.path.abspath(__file__)
mindocr_path = dirname(dirname(dirname(dirname(current_file_path))))
if mindocr_path not in sys.path:
    sys.path.append(mindocr_path)

model_processor = ModelProcessor(os.path.join(dirname(current_file_path), "config.yaml"))


# define preprocess and postprocess
def preprocess(data_nparray: np.ndarray) -> tuple:
    return model_processor.preprocess(data_nparray)


def postprocess(pred: np.ndarray) -> List[np.ndarray]:
    return model_processor.postprocess_method(pred)["texts"]


# register model
model = register.declare_model(model_file="model.mindir", model_format="MindIR", with_batch_dim=False)


def model_infer(net_inputs) -> np.ndarray:
    return model.call(net_inputs)


# register url
@register.register_method(output_names=["texts"])
def infer(image):
    net_inputs = register.add_stage(preprocess, image, outputs_count=1)
    output = register.add_stage(model_infer, net_inputs, outputs_count=1)
    texts = register.add_stage(postprocess, output, outputs_count=1)
    return texts
