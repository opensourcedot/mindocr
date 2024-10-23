# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Client for lenet"""
import argparse
import os
from mindspore_serving.client import Client
import numpy as np


def read_images():
    """Read images for directory test_image"""
    images_buffer = []
    image_files = []
    for path, _, file_list in os.walk("./mytest/"):
        for file_name in file_list[:5]:
            image_file = os.path.join(path, file_name)
            image_files.append(image_file)
    for image_file in image_files:
        with open(image_file, "rb") as fp:
            images_buffer.append(fp.read())
    return images_buffer, image_files


def run_restful_classify_top1(model_name, restful_address):
    """RESTful Client for servable lenet and method classify_top1"""
    print("run_restful_classify_top1-----------")
    import base64
    import requests
    import json
    instances = []
    images_buffer, image_files = read_images()
    for image in images_buffer:
        base64_data = base64.b64encode(image).decode()
        instances.append({"image": {"b64": base64_data}})
    instances_map = {"instances": instances}
    post_payload = json.dumps(instances_map)
    ip = "localhost"
    restful_port = int(restful_address.split(":")[-1])
    servable_name = model_name
    method_name = "det_infer"
    result = requests.post(f"http://{ip}:{restful_port}/model/{servable_name}:{method_name}", data=post_payload)
    result = json.loads(result.text)
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name",
                        help="for example: east_mobilenetv3_icdar15")
    parser.add_argument("restful_address",
                        default="127.0.0.1:1500")
    args = parser.parse_args()
    run_restful_classify_top1(args.model_name, args.restful_address)
