import glob
import json
import os

from shapely.geometry import Polygon


class MTWI2018_Converter(object):
    """
    Format annotation to standard form for MTWI2018 dataset.
    """

    def __init__(self, path_mode="relative", **kwargs):
        self.path_mode = path_mode

    def convert(self, task="det", image_dir=None, label_path=None, output_path=None):
        self.label_path = label_path
        assert os.path.exists(label_path), f"{label_path} no exist!"

        if task == "det":
            self._format_det_label(image_dir, self.label_path, output_path)
        else:
            raise ValueError("MTWI2018 currently only support detection.")

    def _format_det_label(self, image_dir, label_dir, output_path):
        label_paths = sorted(glob.glob(os.path.join(label_dir, "*.txt")))

        processed = 0
        with open(output_path, "w") as out_file:
            for label_fp in label_paths:
                label_file_name = os.path.basename(label_fp)
                img_path = os.path.join(image_dir, label_file_name.split(".txt")[0] + ".jpg")
                assert os.path.exists(
                    img_path
                ), f"{img_path} not exist! Please check the input image_dir {image_dir} and names in {label_fp}"
                label = []
                if self.path_mode == "relative":
                    img_path = os.path.basename(img_path)
                with open(label_fp, "r", encoding="utf-8-sig") as f:
                    for line in f.readlines():
                        tmp = line.strip("\n\r").split(",")
                        points = [[float(tmp[i]), float(tmp[i + 1])] for i in range(0, 8, 2)]

                        if not Polygon(points).exterior.is_ccw:  # sort vertices in polygons in clockwise order
                            points = points[::-1]
                        transcription = tmp[8:]
                        if len(transcription) != 1:
                            transcription = ",".join(transcription)
                        else:
                            transcription = transcription[0]
                        result = {"transcription": transcription, "points": points}
                        label.append(result)

                out_file.write(img_path + "\t" + json.dumps(label, ensure_ascii=False) + "\n")
                processed += 1
            print(f"processed {processed} images.")
