from pathlib import Path

from mindocr import DetModel, RecModel

cur_dir = Path(__file__).resolve().parent
test_file_dir = cur_dir / "test_engine_cfg"

# This specifies the address of the data set used by the test case, which runs the test directly
CFG_FILE = str(test_file_dir / "crnn_resnet34.yaml")
IMG_PATHS = str(test_file_dir / "CRNN.png")


def test_detmodel_default_cfg():
    casemodel = DetModel(algo="DB++")
    res = casemodel.infer(IMG_PATHS)
    print(res)


def test_recmodel_default_cfg():
    casemodel = RecModel(algo="RARE")
    res = casemodel.infer([IMG_PATHS])
    print(res)


def test_recmodel_get_model():
    casemodel = RecModel(algo="RARE")
    getmodel = casemodel.get_model()
    print(getmodel)


def test_infer_from_yaml():
    casemodel = RecModel(init_with_config_file=True, config_file_path=CFG_FILE)
    res = casemodel.infer([IMG_PATHS])
    print(res)


if __name__ == "__main__":
    test_detmodel_default_cfg()
    test_recmodel_default_cfg()
    test_recmodel_get_model()
    test_infer_from_yaml()
