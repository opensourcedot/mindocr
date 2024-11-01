import os
from typing import List


def get_base_path() -> str:
    """
    get mindocr repo base path
    Returns:
        repo base path
    """
    current_path = os.getcwd()
    while not os.path.isdir(os.path.join(current_path, "tests")):
        current_path = os.path.dirname(current_path)
    return current_path


def get_k_folder_name(path: str, k: int) -> str:
    """
    get the name of the k-level directory above the current directory
    Args:
        path: current path
        k: k-level

    Returns:
        the name of the k-level directory above the current directory
    """
    path = os.path.dirname(path)
    count = 0
    full_path = []
    while count < k:
        if not os.path.basename(path):
            break
        full_path.append(os.path.basename(path))
        path = os.path.dirname(path)
        count += 1
    return "***".join(full_path[::-1])


def bfs_search_specific_type_file(root_path: str, file_type: str) -> List[str]:
    """
    use bfs method to find yaml files in root_path
    Args:
        root_path: root_path to search yaml files
        file_type: specific file type to search

    Returns:
        list of specific file path
    """
    specific_file_paths = []

    def bfs_helper(path: str):
        if not os.listdir(path):
            return
        current_level_files = os.listdir(path)
        current_level_file_paths = [os.path.join(path, file) for file in current_level_files]
        for cur_path in current_level_file_paths:
            if os.path.isfile(cur_path):
                if cur_path.endswith(file_type):
                    specific_file_paths.append(cur_path)
            else:
                bfs_helper(cur_path)

    bfs_helper(root_path)

    return specific_file_paths
