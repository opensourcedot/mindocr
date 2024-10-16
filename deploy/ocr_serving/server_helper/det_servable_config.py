import numpy as np

from .model_process_helper import ModelProcessor
from mindspore_serving.server import register
from mindocr.data.transforms import run_transforms

model_processor = ModelProcessor("./config.yaml")


# define preprocess and postprocess
def preprocess(data_nparray: np.ndarray):
    data = {"image": data_nparray, "image_ori": data_nparray.copy(), "image_shape": data_nparray.shape}
    # the input is nparray format, so we don't need to decode
    return run_transforms(data, model_processor.preprocess_method[1:])


def postprocess(pred: dict):
    shape_list = np.array(pred["shape_list"], dtype="float32")
    shape_list = np.expand_dims(shape_list, axis=0)

    output = model_processor.postprocess_method(pred, shape_list=shape_list)

    if isinstance(output, dict):
        polys = output["polys"][0]
        scores = output["scores"][0]
    else:
        polys, scores = output[0]

    return dict(polys=polys, scores=scores)


# register model
model = register.declare_model(model_file="model.mindir", model_format="MindIR", with_batch_dim=True)


# register url
@register.register_method(output_names=["result"])
def det_infer(image):
    x = register.add_stage(preprocess, image)
    x = register.add_stage(model, x)
    x = register.add_stage(postprocess, x)
    return x