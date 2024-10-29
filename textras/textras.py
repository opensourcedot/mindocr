from . import layout, text, utility
from .layout import LayoutAnalyzer, visualize
from .text import TextSystem
from .utility import add_padding, sort_words_by_poly

from tools.infer.text.config import parse_args

__all__ = ["Textras"]

class Textras(object):
    '''
    Textras is a tool for text recognition and layout analysis

    Attributes:
        ocr: text detection and recognition system
        layout: layout analysis system

    Methods:
        __init__(self, ocr=True, layout=True): Initialize Textras
    '''
    def __init__(self, **kargs):
        '''
        Initialize Textras
        '''
        args = parse_args()
        for k, v in kargs.items():
            setattr(args, k, v)
        self.ocr = self.init_ocr(args)
        self.layout = self.init_layout(args)
    
    def analyze(self, img_path, output_path=None):
        '''
        Analyze a paper image and turn it into a structured format 

        Args:
            img_or_path: str for img path or np.array for RGB image
        '''
        image, cls_data = self.layout.infer(img_path)
        category_dict = {1: 'text', 2: 'title', 3: 'list', 4: 'table', 5: 'figure'}
        h_ori, w_ori = image.shape[:2]
        crops = []
        text_results = []
        for i in range(len(cls_data)):
            category_id = cls_data[i]['category_id']
            left, top, w, h = cls_data[i]['bbox']
            right = left + w
            bottom = top + h

            cropped_img = image[int(top):int(bottom), int(left):int(right)]
            cropped_img = add_padding(cropped_img, padding_size=10, padding_color=(255, 255, 255))
            crops.append(cropped_img)
            rec_res_all_crops = text_system(cropped_img, do_visualize=False)
            output = sort_words_by_poly(rec_res_all_crops[1], rec_res_all_crops[0])
            text_results.append({"category_id": category_id, "bbox": [left, top, w, h], "text": " ".join(output)})

        return text_results

    def init_ocr(self, args):
        '''
        Initialize text detection and recognition system

        Args:
            ocr: enable text system or not
            det_algorithm: detection algorithm
            rec_algorithm: recognition algorithm
            det_model_dir: detection model directory
            rec_model_dir: recognition model directory
        '''
        if args.ocr:
            return TextSystem(args)
        return None 

    def init_layout(self, args):
        '''
        Initialize layout analysis system

        Args:
            layout: enable layout module or not
            layout_model_dir: layout model ckpt path
        '''
        if hasattr(args, "layout") and args.layout:
            return LayoutAnalyzer(args)
        return None

