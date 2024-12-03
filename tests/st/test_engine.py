import sys
from pathlib import Path

sys.path.append(".")
from mindocr.engine import DetModel, RecModel

cur_dir = Path(__file__).resolve().parent
test_file_dir = cur_dir / "test_engine_cfg"

# This specifies the address of the data set used by the test case, which runs the test directly
CFG_FILE = str(test_file_dir / "crnn_resnet34.yaml")
IMG_PATHS = str(test_file_dir / "CRNN.png")


def test_detmodel_default_cfg_for_DBPP():
    casemodel = DetModel(algo="DB++")
    res = casemodel.infer(IMG_PATHS)
    print(res)


def test_detmodel_default_cfg_for_DB():
    casemodel = DetModel(algo="DB")
    res = casemodel.infer(IMG_PATHS)
    print(res)


def test_detmodel_default_cfg_for_DB_MV3():
    casemodel = DetModel(algo="DB_MV3")
    res = casemodel.infer(IMG_PATHS)
    print(res)


def test_detmodel_default_cfg_for_DB_PPOCRv3():
    casemodel = DetModel(algo="DB_PPOCRv3")
    res = casemodel.infer(IMG_PATHS)
    print(res)


def test_detmodel_default_cfg_for_PSE():
    casemodel = DetModel(algo="PSE")
    res = casemodel.infer(IMG_PATHS)
    print(res)


def test_recmodel_default_cfg_for_CRNN():
    casemodel = RecModel(algo="CRNN")
    res = casemodel.infer([IMG_PATHS])
    print(res)


def test_recmodel_default_cfg_for_RARE():
    casemodel = RecModel(algo="RARE")
    res = casemodel.infer([IMG_PATHS])
    print(res)


def test_recmodel_default_cfg_for_CRNN_CH():
    casemodel = RecModel(algo="CRNN_CH")
    res = casemodel.infer([IMG_PATHS])
    print(res)


def test_recmodel_default_cfg_for_RARE_CH():
    casemodel = RecModel(algo="RARE_CH")
    res = casemodel.infer([IMG_PATHS])
    print(res)


def test_recmodel_default_cfg_for_SVTR():
    casemodel = RecModel(algo="SVTR")
    res = casemodel.infer([IMG_PATHS])
    print(res)


def test_recmodel_default_cfg_for_SVTR_PPOCRv3_CH():
    casemodel = RecModel(algo="SVTR_PPOCRv3_CH")
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
    # Detection model testing
    test_detmodel_default_cfg_for_DBPP()
    test_detmodel_default_cfg_for_DB()
    test_detmodel_default_cfg_for_DB_MV3()
    test_detmodel_default_cfg_for_DB_PPOCRv3()
    test_detmodel_default_cfg_for_PSE()
    # Recognition model testing
    test_recmodel_default_cfg_for_CRNN()
    test_recmodel_default_cfg_for_RARE()
    test_recmodel_default_cfg_for_CRNN_CH()
    test_recmodel_default_cfg_for_RARE_CH()
    test_recmodel_default_cfg_for_SVTR()
    test_recmodel_default_cfg_for_SVTR_PPOCRv3_CH()
    # Functional interface testing
    test_recmodel_get_model()
    test_infer_from_yaml()
