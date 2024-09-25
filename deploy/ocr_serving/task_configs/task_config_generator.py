import os
import re
import time

import yaml

from deploy.ocr_serving.package_utils.path_utils import get_base_path, bfs_search_specific_type_file


class TaskConfigGenerator:
    def __get_key_information_from_yaml(self, yaml_file_path: str) -> dict:
        """

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
                           "name": os.path.basename(yaml_file_path).split(".yaml")[0]}
        return config_dict

    def __get_key_information_from_markdown(self, markdown_file: str) -> dict:
        pattern_ckpt = r'\[ckpt\]\((https://[^\)]+)\)'
        pattern_mindir = r'\[mindir\]\((https://[^\)]+)\)'
        pattern_yaml = r'\(([^\)]+)\.yaml\)'
        result_dict = dict()
        with open(markdown_file, "r", encoding="utf-8") as f:
            for line in f:
                cur_ckpt_file_link = re.findall(pattern_ckpt, line)
                cur_mindir_file_link = re.findall(pattern_mindir, line)
                if cur_ckpt_file_link or cur_mindir_file_link:
                    model_name = line.split("|")[1].split("|")[0].strip()
                    backbone_name = line.split("|")[3].split("|")[0].strip()
                    identifier = model_name + backbone_name
                    result_dict[identifier] = []
                    result_dict[identifier].append(cur_ckpt_file_link) if cur_ckpt_file_link else result_dict[
                        identifier].append(None)
                    result_dict[identifier].append(cur_mindir_file_link) if cur_mindir_file_link else result_dict[
                        identifier].append(None)
        print(result_dict)

    def combine_all_yaml_files_key_info(self):
        """
        get all_configs.yaml
        Returns:
        """
        base_path = get_base_path()

        configs_base_path = os.path.join(base_path, "configs")

        final_result = []
        config_yaml_files = bfs_search_specific_type_file(configs_base_path, ".yaml")
        markdown_files = bfs_search_specific_type_file(configs_base_path, "README.md")
        for yaml_file in config_yaml_files:
            final_result.append(self.__get_key_information_from_yaml(yaml_file))
        for markdown_file in markdown_files:
            self.__get_key_information_from_markdown(markdown_file)
        with open("all_configs.yaml", "w+", encoding="utf-8") as f:
            yaml.dump_all(documents=final_result, stream=f, allow_unicode=True)


if __name__ == "__main__":
    task_config_generator = TaskConfigGenerator()
    task_config_generator.combine_all_yaml_files_key_info()
