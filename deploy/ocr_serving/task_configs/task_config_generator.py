import os
import re

import yaml

from deploy.ocr_serving.package_utils.path_utils import get_base_path, bfs_search_specific_type_file

COMBINED_CONFIG_YAML_FILE_NAME = "combined_configs.yaml"
MODELS_LINK_PATH = "docs/zh/inference/inference_quickstart.md"


def get_key_information_from_yaml(yaml_file_path: str) -> dict:
    """
    get postprocess, preprocess, yaml_file_name from yanl
    Args:
        yaml_file_path: yaml_file_path

    Returns:
        full key information of all yaml files in configs of mindocr
    """
    with open(yaml_file_path, "r", encoding="utf-8") as f:
        result = yaml.load(f.read(), Loader=yaml.FullLoader)
        post_process = result["postprocess"] if "postprocess" in result.keys() else None
        pre_process = result["eval"]["dataset"]["transform_pipeline"] if "eval" in result.keys() else None
        config_dict = {"postprocess": post_process, "preprocess": pre_process, "use_pretrained_mindir": True,
                       "yaml_file_name": os.path.basename(yaml_file_path)}
    return config_dict


class TaskConfigGenerator:
    def __init__(self):
        self.__mindocr_base_path = get_base_path()
        self.__models_link_path = os.path.join(self.__mindocr_base_path, MODELS_LINK_PATH)
        self.__models_link = []
        self.__config_yaml_files = []
        self.__model_configs = []

    def __get_models_links(self):
        """
        from markdown file to get models links
        Returns:
        """
        path = os.path.join(self.__mindocr_base_path, MODELS_LINK_PATH)

        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                # 正则表达式模式，用于匹配data shape (NCHW)、ckpt和yaml的下载链接
                pattern = (r'\((\d+,\d+,\d+,\d+)\)\s+\|.*?\[yaml\]\((https://github\.com[^\)]+)\).*?\[ckpt\]\(('
                           r'https://download\.mindspore\.cn[^\)]+)\).*?\[mindir\]\((https://download\.mindspore\.cn['
                           r'^\)]+)\)')

                # 使用正则表达式找到所有匹配的数据
                matches = re.findall(pattern, line, re.DOTALL)

                # 打印结果
                for match in matches:
                    info_dict = {"data_shape_nchw": [int(elem) for elem in match[0].split(",")],
                                 "yaml_file_name": match[1].split(r"/")[-1], "ckpt_link": match[2]}
                    self.__models_link.append(info_dict)

    def __add_model_link_to_model_configs(self):
        """
        add model link(ckpt/ mindir) to model configs
        Returns:
        """
        processed_model_configs = []
        for model_config in self.__model_configs:
            model_config["ckpt_link"] = ""
            model_config["custom_mindir_link"] = ""
            for model_link in self.__models_link:
                if model_config["yaml_file_name"] == model_link["yaml_file_name"]:
                    model_config["ckpt_link"] = model_link["ckpt_link"]
                    model_config["data_shape_nchw"] = model_link["data_shape_nchw"]
                    break
            if not model_config["ckpt_link"]:
                model_config["use_pretrained_mindir"] = False
            processed_model_configs.append(model_config)
        self.__model_configs = processed_model_configs

    def generate_config_file(self):
        """
        get all_configs.yaml
        Returns:
        """
        # 1. set base path (mindocr/ configs)
        configs_base_path = os.path.join(self.__mindocr_base_path, "configs")

        # 2. search config yaml files from base path
        config_yaml_files = bfs_search_specific_type_file(configs_base_path, ".yaml")

        # 3. get model preprocess, postprocess ... from model config yaml files
        for yaml_file in config_yaml_files:
            self.__model_configs.append(get_key_information_from_yaml(yaml_file))

        # 4. get model links (include ckpt link, mindir link)
        self.__get_models_links()

        # 5. add model ckpt/ mindir link to model_configs
        self.__add_model_link_to_model_configs()

        # 6. combine these configs into one config.yaml
        with open("all_configs.yaml", "w+", encoding="utf-8") as f:
            yaml.dump_all(documents=self.__model_configs, stream=f, allow_unicode=True)


if __name__ == "__main__":
    task_config_generator = TaskConfigGenerator()
    task_config_generator.generate_config_file()
