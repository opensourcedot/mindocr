import os
import sys
from os.path import dirname

import numpy as np
from mindspore_serving.server import register

from .model_process_helper import ModelProcessor

current_file_path = os.path.abspath(__file__)
mindocr_path = dirname(dirname(dirname(dirname(current_file_path))))
if mindocr_path not in sys.path:
    sys.path.append(mindocr_path)

model_processor = ModelProcessor(os.path.join(dirname(current_file_path), "config.yaml"))


# define preprocess and postprocess
def preprocess(data_nparray: np.ndarray):
    return model_processor.preprocess_method([data_nparray])


def postprocess(pred: dict):
    return model_processor.postprocess_method()


# register model
model = register.declare_model(model_file="model.mindir", model_format="MindIR", with_batch_dim=True)


# register url
@register.register_method(output_names=["result"])
def det_infer(image):
    x = register.add_stage(preprocess, image, outputs_count=1)
    x = register.add_stage(model, x, outputs_count=2)
    x = register.add_stage(postprocess, x, outputs_count=1)
    return x
