"""
用户输入不同的模型名称，即可将对应的模型打包至对应的文件夹里
（1）使用自己的 mindir 文件
（2）使用我们提供的 mindir 文件
"""
import os
import shutil

import yaml
import requests

from deploy.ocr_serving.package_utils.path_utils import get_base_path

SUPPORT_INFER_TYPE = ["mindir", "ms"]


class PackageHelper:
    def __init__(self, package_name: str, mindir_file_path: str = None):
        """
        Args:
            package_name: the model name we want to package
            mindir_file_path: custom mindir file path. default None
        """
        self.package_name = package_name
        self.mindir_path = mindir_file_path
        self.use_custom_mindir = False
        # input check
        self.input_check()
        # mindocr base path
        self.base_path = get_base_path()
        # target model server folder
        self.target_server_infer_folder = None
        # mindir file link
        self.mindir_file_link = None
        # target mindir file folder
        self.target_mindir_folder = None

    def input_check(self):
        """
        check weather the input mindir_file_path is valid
        Returns:
        """
        if self.mindir_path is None:
            pass
        else:
            self.use_custom_mindir = True
            model_type = self.mindir_path.split(".")[-1]
            if model_type not in SUPPORT_INFER_TYPE:
                raise Exception("unsupport mindir file type :{}".format(model_type))
            # if chose to use custom mindir file
            if not os.path.exists(self.mindir_path):
                raise Exception("mindir file not exist at : {}".format(self.mindir_path))

    def build_infer_server_folder(self):
        """
        build mindspore serving inference folder
        Returns:
        """
        base_server_folder = os.path.join(self.base_path, "deploy/ocr_serving/server_folders")
        self.target_server_infer_folder = os.path.join(self.base_path, "deploy/ocr_serving/server_folders", self.package_name)
        # build the base folder if not exists
        if os.path.exists(base_server_folder):
            pass
        else:
            os.mkdir(base_server_folder)
        # build the target inference server folder
        if os.path.exists(self.target_server_infer_folder):
            pass
        else:
            os.mkdir(self.target_server_infer_folder)

    def get_target_config_yaml(self):
        """
        get target config yaml from all_configs.yaml
        Returns:
        """
        base_config_yaml_path = os.path.join(self.base_path, "deploy/ocr_serving/task_configs/all_configs.yaml")
        target_config_yaml_path = os.path.join(self.target_server_infer_folder, self.package_name + ".yaml")

        matched = False
        # read configs
        with open(base_config_yaml_path, "r", encoding="utf-8") as f:
            yaml_configs = yaml.load_all(f.read(), Loader=yaml.FullLoader)
            all_configs = [yaml_config for yaml_config in yaml_configs]

        # match and get target config
        for yaml_config in all_configs:
            if yaml_config["yaml_file_name"].replace(".yaml", "").lower() == self.package_name.lower():
                matched = True
                if self.use_custom_mindir:
                    yaml_config["use_pretrained_mindir"] = False
                    yaml_config["mindir_link"] = self.mindir_path
                if not yaml_config["mindir_link"]:
                    raise Exception("No valid mindir file")
                self.mindir_file_link = yaml_config["mindir_link"]
                with open(target_config_yaml_path, "w+", encoding="utf-8") as g:
                    yaml.dump(yaml_config, stream=g)
                break

        if not matched:
            raise Exception("model not matched in all_configs.yaml")

    def get_mindir_file(self):
        """
        if use official mindir file, we need to download mindir file.
        if use custom mindir file, we need to copy the mindir file to target folder
        Returns:
        """
        dirs = os.listdir(self.target_server_infer_folder)
        dir_list = [0]
        for dir in dirs:
            if os.path.isdir(os.path.join(self.target_server_infer_folder, dir)):
                try:
                    dir_list.append(int(dir))
                except:
                    pass
        max_index = max(dir_list)
        self.target_mindir_folder = os.path.join(self.target_server_infer_folder, "{}".format(max_index + 1))

        os.mkdir(self.target_mindir_folder)
        # use official mindir file
        if not self.use_custom_mindir:
            self.download_official_mindir_file()
        else:
            self.copy_custom_mindir_file()

    def download_official_mindir_file(self):
        """
        download official mindir file
        Returns:
        """
        response = requests.get(self.mindir_file_link)
        if response.status_code % 100 == 2:
            raise Exception(
                "download mindir official mindir file failed. with status code = {}".format(response.status_code))
        else:
            with open(os.path.join(self.target_mindir_folder, "model.mindir"), "wb") as f:
                f.write(response.content)

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
        dst_path = os.path.join(self.target_server_infer_folder, "/model_process_helper.py")
        if os.path.exists(dst_path):
            shutil.rmtree(dst_path)
        shutil.copy(os.path.join(self.base_path, "deploy/ocr_serving/server_helper/model_process_helper.py"),
                    self.target_server_infer_folder)

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


if __name__ == "__main__":
    package_helper = PackageHelper("east_mobilenetv3_icdar15", r"D:\model.mindir")
    package_helper.do_package()

