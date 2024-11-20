import logging
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../../../")))

from tools.infer.text.config import parse_args
from tools.infer.text.predict_system import TextClassifier
from tools.infer.text.utils import get_image_paths

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

def test_cls_infer():
    args = parse_args()
    logger = logging.getLogger(__name__)
    text_classification = TextClassifier(args)
    if os.path.isfile(args.image_dir):
        cls_res_all = text_classification(args.image_dir)
    else:
        img_paths = get_image_paths(args.image_dir)
        cls_res_all = text_classification(img_paths)
    logger.info(f"All cls res: {cls_res_all}")


if __name__ == "__main__":
    test_cls_infer()
