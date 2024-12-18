import gradio as gr
import os
from predict_layout import predict_layout
from predict_system import predict_system
from predict_table_recognition import predict_table
from predict_table_e2e import predict_e2e
from predict_e2e import predict_rec_e2e
from predict_ser import predict_ser

def process_image(image, mode):
    save_dir = "inference_results"
    os.makedirs(save_dir, exist_ok=True)
    img_name = os.path.basename(image).rsplit(".", 1)[0]

    if mode == "版面识别":
        save_path = os.path.join(save_dir, "layout_res.png")
        predict_layout(image, save_dir)
        return save_path, None
    elif mode == "文字识别(DB++/CRNN)":
        save_path = os.path.join(save_dir, "text_ocr_res.png")
        predict_system(image, save_dir)
        return save_path, None
    elif mode == "文字识别(PGNet)":
        save_path = os.path.join(save_dir, "text_e2e_ocr_res.png")
        predict_rec_e2e(image, save_dir)
        return save_path, None
    elif mode == "表格识别":
        save_path = os.path.join(save_dir, "table_reconization.csv")
        predict_table(image, save_dir)
        return None, save_path
    elif mode == "关键信息抽取":
        save_path = os.path.join(save_dir, "kie_ocr_res.png")
        predict_ser(image, save_dir)
        return None, save_path
    elif mode == "文档转换":
        save_path = os.path.join(save_dir, "e2e.docx")
        predict_e2e(image, save_dir)
        return None, save_path

    return None, None

def handle_mode_change(mode):
    if mode == "表格识别" or mode == "文档转换":
        return gr.update(visible=False), gr.update(visible=True)
    else:
        return gr.update(visible=True), gr.update(visible=False)

# 使用 Blocks 构建界面
with gr.Blocks() as demo:
    gr.Markdown("# MindOCR端到端文字识别套件")
    with gr.Row():
        image_input = gr.Image(type="filepath", label="上传图片")
        mode_selector = gr.Radio(["版面识别", "文字识别(DB++/CRNN)", "文字识别(PGNet)", "表格识别", "关键信息抽取", "文档转换"], label="选择OCR模式", interactive=True)

    with gr.Row():
        result_image = gr.Image(type="filepath", label="处理结果", visible=True)
        result_file = gr.File(label="处理结果", visible=False)

    mode_selector.change(handle_mode_change, inputs=[mode_selector], outputs=[result_image, result_file])

    submit_button = gr.Button("开始处理")
    submit_button.click(
        fn=process_image,
        inputs=[image_input, mode_selector],
        outputs=[result_image, result_file],
    )

demo.launch()