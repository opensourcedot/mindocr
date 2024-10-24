DET_METHODS_SET = {
    "db_mobilenetv3_icdar15.yaml",
    "db_r18_ctw1500.yaml",
    "db_r18_icdar15.yaml",
    "db_r18_mlt2017.yaml",
    "db_r18_synthtext.yaml",
    "db_r18_td500.yaml",
    "db_r18_totaltext.yaml",
    "dbpp_r50_icdar15.yaml",
    "db_r50_ctw1500.yaml",
    "db_r50_icdar15.yaml",
    "db_r50_mlt2017.yaml",
    "db_r50_synthtext.yaml",
    "db_r50_td500.yaml",
    "db_r50_totaltext.yaml",
    "east_mobilenetv3_icdar15.yaml",
    "east_r50_icdar15.yaml",
    "fce_icdar15.yaml",
    "pse_mv3_icdar15.yaml",
    "pse_r152_ctw1500.yaml",
    "pse_r152_icdar15.yaml",
    "pse_r50_icdar15.yaml"
}

CLS_METHODS_SET = {
    "cls_mv3.yaml"
}

KIE_METHODS_SET = {
    "ser_layoutxlm_xfund_zh.yaml"
}

REC_METHODS_SET = {
    "abinet_resnet45_en.yaml",
    "crnn_resnet34.yaml",
    "crnn_resnet34_ch.yaml",
    "crnn_vgg7.yaml",
    "master_resnet31.yaml",
    "rare_resnet34.yaml",
    "rare_resnet34_ch.yaml",
    "robustscanner_resnet31.yaml",
    "svtr_ppocrv3_ch.yaml",
    "svtr_tiny.yaml",
    "svtr_tiny_ch.yaml"
    "visionlan_resnet45_LA.yaml",
    "visionlan_resnet45_LF_1.yaml",
    "visionlan_resnet45_LF_2.yaml"
}

ALL_TASK_TYPE_DICT = {
    "det": DET_METHODS_SET,
    "cls": CLS_METHODS_SET,
    "kie": KIE_METHODS_SET,
    "rec": REC_METHODS_SET
}

EXPORT_NAME_MAPPER = {
    "cls_mv3.yaml": "cls_mobilenet_v3_small_100_model",
    "abinet_resnet45_en.yaml": "abinet",
    "crnn_resnet34.yaml": "crnn_resnet34",
    "crnn_resnet34_ch.yaml": "crnn_resnet34_ch",
    "crnn_vgg7.yaml": "crnn_vgg7",
    "db_mobilenetv3_icdar15.yaml": "dbnet_mobilenetv3",
    "db_r18_ctw1500.yaml": "dbnet_resnet18",
    "db_r18_icdar15.yaml": "dbnet_resnet18",
    "db_r18_mlt2017.yaml": "dbnet_resnet18",
    "db_r18_synthtext.yaml": "dbnet_resnet18",
    "db_r18_td500.yaml": "dbnet_resnet18",
    "db_r18_totaltext.yaml": "dbnet_resnet18",
    "dbpp_r50_icdar15.yaml": "dbnetpp_resnet50",
    "db_r50_ctw1500.yaml": "dbnet_resnet50",
    "db_r50_icdar15.yaml": "dbnet_resnet50",
    "db_r50_mlt2017.yaml": "dbnet_resnet50",
    "db_r50_synthtext.yaml": "dbnet_resnet50",
    "db_r50_td500.yaml": "dbnet_resnet50",
    "db_r50_totaltext.yaml": "dbnet_resnet50",
    "east_mobilenetv3_icdar15.yaml": "east_mobilenetv3",
    "east_r50_icdar15.yaml": "east_resnet50",
    "fce_icdar15.yaml": "fcenet_resnet50",
    "pse_mv3_icdar15.yaml": "psenet_mobilenetv3",
    "pse_r152_ctw1500.yaml": "psenet_resnet152",
    "pse_r152_icdar15.yaml": "psenet_resnet152",
    "pse_r50_icdar15.yaml": "psenet_resnet50",
    "ser_layoutxlm_xfund_zh.yaml": "layoutxlm_ser",
    "master_resnet31.yaml": "master_resnet31",
    "rare_resnet34.yaml": "rare_resnet34",
    "rare_resnet34_ch.yaml": "rare_resnet34_ch",
    "robustscanner_resnet31.yaml": "robustscanner_resnet31",
    "svtr_ppocrv3_ch.yaml": "svtr_ppocrv3_ch",
    "svtr_tiny.yaml": "svtr_tiny",
    "svtr_tiny_ch.yaml": "svtr_tiny_ch",
    "visionlan_resnet45_LA.yaml": "visionlan_resnet45",
    "visionlan_resnet45_LF_1.yaml": "visionlan_resnet45",
    "visionlan_resnet45_LF_2.yaml": "visionlan_resnet45"
}
