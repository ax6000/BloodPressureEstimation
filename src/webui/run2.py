import sys
sys.path.append(r"F:\minowa\BloodPressureEstimation\.venv\Lib\site-packages")
import gradio as gr
import glob
import os

def glob_expl():
    path = r"F:\minowa\BloodPressureEstimation\data\raw\ppgabp\*"
    # return gr.FileExplorer(glob=path,height=250)
    return a

def glob_expl2():
    path = r"F:\minowa\BloodPressureEstimation\data\raw\ppgabp\*"
    # return gr.FileExplorer(glob=path,height=250)
    return gr.FileExplorer(glob=path,height=250)

default_path = r"F:\data\*"
with gr.Blocks() as demo:
    with gr.Row():
        data_button = gr.Button("Update")
        data_files = gr.FileExplorer(glob=default_path,height=300)
    data_button.click(glob_expl,outputs=data_files)


if __name__ == "__main__":
    demo.launch()