"""
用户输入不同的模型名称，即可将对应的模型打包至对应的文件夹里
（1）使用自己的 mindir 文件
（2）使用我们提供的 mindir 文件
"""
import argparse
import math
import os
import shutil
import sys
from typing import Dict

import requests
import yaml
from package_utils.mappers import EXPORT_NAME_MAPPER, SERVABLE_CONFIGS_MAPPER
from package_utils.path_utils import get_base_path

current_file_path = os.path.abspath(__file__)
mindocr_path = os.path.dirname(os.path.dirname(current_file_path))

if mindocr_path not in sys.path:
    sys.path.append(mindocr_path)

SUPPORT_INFER_TYPE = ["mindir", "ms"]
TARGET_SERVER_FOLDER = "deploy/ocr_serving/server_folders"
ALL_CONFIGS_YAML_PATH = os.path.join(get_base_path(), "deploy/ocr_serving/task_configs/all_configs.yaml")
CORRECT_RESPONSE_CODE = 2


class PackageException(Exception):
    def __init__(self, message="Package Error."):
        super().__init__(message)


class PackageHelper:
    def __init__(
        self, package_name: str, mindir_file_path: str = None, test_mode: bool = True, ch_detection: bool = False
    ):
        """
        Args:
            package_name: the model name we want to package
            mindir_file_path: custom mindir file path. default None
            test_mode: True if you need to do test
            ch_detection: weather chinese ocr detection
        """
        self.package_name = package_name
        self.custom_mindir_path = mindir_file_path
        self.use_custom_mindir = False
        # input check
        self.input_check()
        # mindocr base path
        self.base_path = get_base_path()
        # target mofrl folder
        self.target_model_folder = None
        # target server folder
        self.target_server_folder = TARGET_SERVER_FOLDER
        # mindir file link
        self.mindir_file_link = None
        # target mindir file folder
        self.target_mindir_folder = None
        # target config yaml
        self.target_config_yaml = None
        self.test_mode = test_mode
        self.ch_detection = ch_detection

    def input_check(self):
        """
        check weather the input mindir_file_path is valid
        Returns:
        """
        # if we not provide custom mindir path
        if self.custom_mindir_path is None:
            pass
        else:
            # use custom provided custom mindir path
            self.use_custom_mindir = True
            # check model type
            model_type = self.custom_mindir_path.split(".")[-1]
            if model_type not in SUPPORT_INFER_TYPE:
                raise PackageException(f"unsupport mindir file type :{model_type}")
            # if chose to use custom mindir file
            if not os.path.exists(self.custom_mindir_path):
                raise PackageException(f"mindir file not exist at : {self.custom_mindir_path}")

    def build_infer_server_folder(self):
        """
        build mindspore serving inference folder
        Returns:
        """
        base_server_folder = os.path.join(self.base_path, TARGET_SERVER_FOLDER)

        self.target_model_folder = os.path.join(self.base_path, self.target_server_folder, self.package_name)
        # build the base folder if not exists
        if os.path.exists(base_server_folder):
            pass
        else:
            os.mkdir(base_server_folder)
        # build the target inference server folder
        if os.path.exists(self.target_model_folder):
            pass
        else:
            os.mkdir(self.target_model_folder)

    def get_target_config_yaml(self):
        """
        get target config yaml from all_configs.yaml
        Returns:
        """
        base_config_yaml_path = ALL_CONFIGS_YAML_PATH
        target_config_yaml_path = os.path.join(self.target_model_folder, "config.yaml")

        matched = False
        # read all configs
        with open(base_config_yaml_path, "r", encoding="utf-8") as f:
            all_configs = [yaml_config for yaml_config in yaml.load_all(f.read(), Loader=yaml.FullLoader)]

        # match and get target config
        for yaml_config in all_configs:
            if yaml_config["yaml_file_name"].replace(".yaml", "").lower() == self.package_name.lower():
                matched = True
                self._update_yaml_configs_for_special_situations(yaml_config)
                # case 1: use custom mindir file
                if self.use_custom_mindir:
                    yaml_config["use_pretrained_mindir"] = False
                    yaml_config["custom_mindir_link"] = self.custom_mindir_path
                # case 2: use official mindir file
                else:
                    if not yaml_config["ckpt_link"]:
                        raise PackageException("No valid ckpt file to convert")
                with open(target_config_yaml_path, "w+", encoding="utf-8") as g:
                    yaml.dump(yaml_config, stream=g)
                self.target_config_yaml = yaml_config
                break

        if not matched:
            raise PackageException("model not matched in all_configs.yaml")

    def _update_yaml_configs_for_special_situations(self, yaml_config: Dict):
        yaml_config = self.__update_character_dict_path(yaml_config)
        yaml_config = self.__change_target_shape(yaml_config)
        return yaml_config

    def __update_character_dict_path(self, yaml_config: Dict) -> Dict:
        if "character_dict_path" in yaml_config["postprocess"] and yaml_config["postprocess"]["character_dict_path"]:
            yaml_config["postprocess"]["character_dict_path"] = os.path.join(
                self.base_path, yaml_config["postprocess"]["character_dict_path"]
            )
        for i, method in enumerate(yaml_config["eval"]["dataset"]["transform_pipeline"]):
            for _, (method_name, method_params) in enumerate(method.items()):
                if (
                    isinstance(method_params, dict)
                    and "character_dict_path" in method_params
                    and method_params["character_dict_path"]
                ):
                    character_dict_path = yaml_config["eval"]["dataset"]["transform_pipeline"][i][method_name][
                        "character_dict_path"
                    ]
                    yaml_config["eval"]["dataset"]["transform_pipeline"][i][method_name][
                        "character_dict_path"
                    ] = os.path.join(self.base_path, character_dict_path)
        return yaml_config

    def __change_target_shape(self, yaml_config: Dict) -> Dict:
        data_shape_nchw = yaml_config["data_shape_nchw"]
        data_shape_nchw = [int(elem) for elem in data_shape_nchw][2:]
        for i, method in enumerate(yaml_config["eval"]["dataset"]["transform_pipeline"]):
            if "DetResize" in method:
                yaml_config["eval"]["dataset"]["transform_pipeline"][i]["DetResize"]["target_size"] = data_shape_nchw
                yaml_config["eval"]["dataset"]["transform_pipeline"][i]["DetResize"]["keep_ratio"] = False
        return yaml_config

    def get_mindir_file(self):
        """
        if use official mindir file, we need to download ckpt file and convert it to mindir file
        if use custom mindir file, we need to copy the mindir file to target folder
        Returns:
        """
        # 1. model version auto increase
        dirs = os.listdir(self.target_model_folder)
        dir_list = [0]
        for dir in dirs:
            if os.path.isdir(os.path.join(self.target_model_folder, dir)):
                try:
                    dir_list.append(int(dir))
                except PackageException:
                    pass
        max_index = max(dir_list)
        self.target_mindir_folder = os.path.join(self.target_model_folder, f"{max_index + 1}")
        os.mkdir(self.target_mindir_folder)

        # 2. get mindir file
        if not self.use_custom_mindir:
            self.get_official_mindir_file()
        else:
            self.copy_custom_mindir_file()

    def get_official_mindir_file(self):
        """
        download official ckpt file and convert it to mindir file
        Returns:
        """
        # 1. get ckpt file from official link
        response = requests.get(self.target_config_yaml["ckpt_link"])
        if math.floor(response.status_code / 100) != CORRECT_RESPONSE_CODE:
            raise PackageException(
                f"download mindir official mindir file failed. with status code = {response.status_code}"
            )
        else:
            target_ckpt_path = os.path.join(self.target_mindir_folder, "model.ckpt")
            with open(target_ckpt_path, "wb") as f:
                f.write(response.content)

        # 2. convert ckpt to mindir
        shell_command = (
            "python {export_tool_path} --model_name_or_config {model_name} "
            "--data_shape {data_shape} --local_ckpt_path {local_ckpt_path} "
            "--save_dir {save_dir} "
            "--custom_exported_name {exported_name}"
        ).format(
            export_tool_path=os.path.join(get_base_path(), "tools/export.py"),
            model_name=EXPORT_NAME_MAPPER[self.target_config_yaml["yaml_file_name"]],
            data_shape=" ".join(self.target_config_yaml["data_shape_nchw"][2:]),
            local_ckpt_path=os.path.join(self.target_mindir_folder, "model.ckpt"),
            save_dir=self.target_mindir_folder,
            exported_name="model",
        )
        os.system(shell_command)
        os.remove(os.path.join(self.target_mindir_folder, "model.ckpt"))

    def copy_custom_mindir_file(self):
        """
        copy custom mindir file to target folder
        Returns:
        """
        shutil.copy(self.mindir_file_link, os.path.join(self.target_mindir_folder, "model.mindir"))

    def copy_model_process_helper(self):
        """
        copy model_process_helper.py to target folder
        Returns:
        """
        src_path = os.path.join(self.base_path, "deploy/ocr_serving/server_helper/model_process_helper.py")
        dst_path = os.path.join(self.target_model_folder, "model_process_helper.py")
        if os.path.exists(dst_path):
            os.remove(dst_path)
        shutil.copy(src_path, dst_path)

    def copy_servable_config(self):
        """
        copy target xxx_servable_config.py to target folder
        Returns:
        """
        src_path = os.path.join(
            self.base_path,
            "deploy/ocr_serving/server_helper/{}.py".format(
                SERVABLE_CONFIGS_MAPPER[self.target_config_yaml["yaml_file_name"]]
            ),
        )
        dst_path = os.path.join(self.target_model_folder, "servable_config.py")
        if os.path.exists(dst_path):
            os.remove(dst_path)
        shutil.copy(src_path, dst_path)

    def copy_serving_server_to_folder(self):
        server_py_src = os.path.join(self.base_path, "deploy/ocr_serving/server_helper/serving_server.py")
        shutil.copy(server_py_src, os.path.join(self.base_path, TARGET_SERVER_FOLDER))

    def copy_test_files(self):
        client_py_src = os.path.join(self.base_path, "deploy/ocr_serving/test/serving_client.py")
        my_test_folder_src = os.path.join(self.base_path, "deploy/ocr_serving/test/mytest")
        target_folder = os.path.join(self.base_path, TARGET_SERVER_FOLDER, "mytest")
        shutil.copytree(my_test_folder_src, target_folder, dirs_exist_ok=True)
        shutil.copy(client_py_src, os.path.join(self.base_path, TARGET_SERVER_FOLDER))

    def do_package(self):
        """
        main function
        Returns:
        """
        # 1. build base folder and target folder
        self.build_infer_server_folder()

        # 2. get target config yaml
        self.get_target_config_yaml()

        # 3. get target mindir file
        self.get_mindir_file()

        # 4. copy model_process_helper.py to folder
        self.copy_model_process_helper()

        # 5. copy servable config to folder
        self.copy_servable_config()

        # 6. copy test serving_server.py and serving_client.py and mytest folder to server_folders
        self.copy_serving_server_to_folder()

        # 7. copy test_files
        if self.test_mode:
            self.copy_test_files()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("package_name", help="for example: east_mobilenetv3_icdar15")
    parser.add_argument(
        "--mindir_file_path", help="if you need to use your local mindir file, please specify this parameter."
    )
    parser.add_argument("--test_mode", help="wheater to switch on test mode", action="store_true")
    parser.add_argument(
        "--ch_det", help="if you need to do chinese ocr detection, you should add this param.", action="store_true"
    )
    args = parser.parse_args()
    package_helper = PackageHelper(args.package_name, args.mindir_file_path, args.test_mode, args.ch_det)
    package_helper.do_package()
