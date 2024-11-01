## 部署步骤

#### 一、安装 mindspore serving

1. 首先安装 mindspore

```shell
pip install mindspore==2.3.1
```

2. clone mindspore serving

```shell
git clone https://gitee.com/mindspore/serving.git -b master
cd serving
```

3. 查看 mindspore 安装路径

```shell
pip show mindspore
>>> Location: /home/ma-user/anaconda3/envs/MindSpore/lib/python3.9/site-packages
```

4. 基于本地的 mindspore 编译，这里的 mindspore_path 即 3 中的安装路径 

```shell
bash build.sh -p ${mindspore_path}/lib -S on
# 如 bash build.sh -p /home/ma-user/anaconda3/envs/MindSpore/lib/python3.9/site-packages/mindspore/lib -S on
```

5. 编译完成后，在`build/package/`目录下找到 whl 安装包进行安装。

```shell
pip install mindspore_serving-{version}-{python_version}-linux_{arch}.whl
```

6. 验证是否安装成功，不报错就成功

```shell
python
>>> from mindspore_serving import server
```

### 二、Clone mindocr 项目进行服务化部署

1. Clone mindocr 

```shell
git clone https://github.com/opensourcedot/mindocr.git
```

2. 安装相关依赖

```shell
cd mindocr
pip install -r requirements.txt
```

3. 进入到 ocr_serving/task_configs 目录

```shell
cd /deploy/ocr_serving/task_configs
```

4. 生成推理配置文件

```shell
python task_config_generator.py
```

5. 返回到 `deploy/ocr_serving` 中执行下述命令，生成 mindspore serving 部署依赖的文件夹到 `deploy/ocr_serving/server_folders`，这里需要输入你所部署的模型的 yaml 文件名称，可以在 configs 下面找到

``` shel
cd ..
python package_utils/package_helper.py {模型名称} --test_mode # 加上 test_mode 会提供测试的脚本。
```

目前 {模型名称} 理论上支持以下输入参数


-  DET 模型
1. db_mobilenetv3_icdar15
2. db_r18_icdar15
3. db_r50_icdar15
4. db_r50_icdar15-ch（失败；未区分中英文）
5. dbpp_r50_icdar15
6. dbpp_r50_icdar15-ch（失败；未区分中英文）
7. east_r50_icdar15
8. east_mobilenetv3_icdar15
9. pse_r152_icdar15
10. pse_r50_icdar15
11. pse_mv3_icdar15
12. fce_icdar15
-  REC 模型
13. crnn_vgg7
14. crnn_resnet34
15. crnn_resnet34_ch
16. svtr_tiny
17. rare_resnet34
18. rare_resnet34_ch
19. robustscanner_resnet31(失败)
20. visionlan_resnet45_LA
- CLS 模型
21. cls_mv3

6. 启动 server 服务

在 `deploy/ocr_serving/server_folders` 下执行

```shell
python serving_server.py {模型名称} {地址}
```

如：

```shell
python serving_server.py db_mobilenetv3_icdar15 127.0.0.1:1500
```

7. 测试

另起一个 kernel ，在 `deploy/ocr_serving/server_folders`  下执行以下命令，注意 {模型名称} 和 {地址} 同执行 serving_server.py 输入的一致

```shell
python serving_server.py {模型名称} {地址}
```



