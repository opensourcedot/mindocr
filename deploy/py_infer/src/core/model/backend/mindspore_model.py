import re
from typing import List

import numpy as np

from .model_base import ModelBase

device_target_mapper = {"ascend": "Ascend", "gpu": "GPU", "cpu": "CPU"}


class MindSporeModel(ModelBase):
    def __init__(self, model_path, device, device_id):
        super().__init__(model_path, device, device_id)
        self.check_input()

    def check_input(self):
        if not isinstance(self.device, str):
            raise Exception("wrong device type : {}".format(type(self.device)))
        if self.device.lower() not in device_target_mapper.keys():
            raise Exception("wrong device : {}".format(self.device))
        if not isinstance(self.device_id, int):
            raise Exception("wrong device id type: {}".format(type(self.device_id)))

    def _init_model(self):
        global ms
        global nn
        global ModelProto
        import mindspore as ms
        from mindspore import nn
        from mindspore.train.mind_ir_pb2 import ModelProto

        # set device
        ms.set_context(device_target=device_target_mapper[self.device.lower()])

        # set device id
        ms.set_context(device_id=self.device_id)

        # build net
        model = ms.load(self.model_path)
        self.net = nn.GraphCell(model)

        # define input shape
        inputs = self.net.get_inputs()
        self._input_num = len(inputs)
        self._input_shape = [x.shape for x in inputs]
        self._input_dtype = [self.__dtype_to_nptype(x.dtype) for x in inputs]

    def infer(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        model_inputs = self.net.get_inputs()
        inputs_shape = [list(input.shape) for input in inputs]
        self.net.resize(model_inputs, inputs_shape)

        for i, input in enumerate(inputs):
            model_inputs[i].set_data_from_numpy(input)

        model_outputs = self.net.predict(model_inputs)
        outputs = [output.get_data_to_numpy().copy() for output in model_outputs]
        return outputs

    def get_gear(self):
        # Only support shape gear for Ascend device.
        if self.device.lower() != "ascend":
            return []

        gears = []

        # MSLite does not provide API to get gear value, so we parse it from origin file.
        with open(self.model_path, "rb") as f:
            content = f.read()

        matched = re.search(rb"_all_origin_gears_inputs.*?\xa0", content, flags=re.S)

        # TODO: shape gear don't support for multi input
        if self._input_num > 1 and matched:
            raise ValueError(
                f"Shape gear don‘t support model input_num > 1 currently, \
                but got input_num = {self._input_num} for {self.model_path}!"
            )

        if not matched:
            return gears

        # TODO: only support NCHW format for shape gear
        matched_text = matched.group()
        shape_text = re.findall(rb"(?<=:4:)\d+,\d+,\d+,\d+", matched_text)

        if not shape_text:
            raise ValueError(
                f"Get gear value failed for {self.model_path}. Please Check converter_lite conversion process!"
            )

        for text in shape_text:
            gear = [int(x) for x in text.decode(encoding="utf-8").split(",")]
            gears.append(gear)

        return gears

    def __dtype_to_nptype(self, type_):

        return {
            ms.dtype.bool_: np.bool_,
            ms.int8: np.int8,
            ms.int16: np.int16,
            ms.int32: np.int32,
            ms.int64: np.int64,
            ms.uint8: np.uint8,
            ms.uint16: np.uint16,
            ms.uint32: np.uint32,
            ms.uint64: np.uint64,
            ms.float16: np.float16,
            ms.float32: np.float32,
            ms.float64: np.float64,
        }[type_]

    def _read_mindir_input_info(self):
        model = ModelProto()
        with open(self.model_path, "rb") as f:
            pb_content = f.read()
            model.ParseFromString(pb_content)
        for item in model.input:
            tensor = item.tensor
            for elem in tensor:
                print(elem.dims)


