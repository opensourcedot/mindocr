import os
import sys
import cv2
import numpy as np

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, "../..")))

from textras import Textras

if __name__ == "__main__":
    t = Textras(ocr=True, layout=True)

