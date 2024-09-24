import os
import sys


def get_base_path():
    current_path = os.getcwd()
    while not os.path.isdir(os.path.join(current_path, "tests")):
        current_path = os.path.dirname(current_path)
    return current_path


def get_k_folder_name(path: str, k: int):
    path = os.path.dirname(path)
    count = 0
    full_path = []
    while count < k:
        full_path.append(os.path.basename(path))
        path = os.path.dirname(path)
        count += 1
    return "--".join(full_path[::-1])


def bfs_config_yamls(root_path: str):
    yaml_file_paths = []

    def bfs_helper(path: str):
        if not os.listdir(path):
            return
        current_level_files = os.listdir(path)
        current_level_file_paths = [os.path.join(path, file) for file in current_level_files]
        for cur_path in current_level_file_paths:
            if os.path.isfile(cur_path):
                if cur_path.endswith(".yaml"):
                    yaml_file_paths.append((cur_path, get_k_folder_name(cur_path, 2)))
            else:
                bfs_helper(cur_path)

    bfs_helper(root_path)

    return yaml_file_paths



BASE_PATH = get_base_path()

configs_base_path = os.path.join(BASE_PATH, "configs")

print(bfs_config_yamls(configs_base_path))
print(len(bfs_config_yamls(configs_base_path)))
