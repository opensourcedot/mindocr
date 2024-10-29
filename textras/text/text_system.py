import sys
import os
import logging

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, "../..")))

from tools.infer.text.predict_system import TextSystem
from tools.infer.text.predict_det import TextDetector
from tools.infer.text.predict_rec import TextRecognizer

from mindocr.utils.logger import set_logger
logger = logging.getLogger("Textras")

__all__ = ["TextSystem", "TextDetector", "TextRecognizer"]
