"""Client for lenet"""
import os


def read_images():
    """Read images for directory test_image"""
    images_buffer = []
    image_files = []
    for path, _, file_list in os.walk("mytest/"):
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