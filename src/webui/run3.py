import sys
sys.path.append(r"D:\minowa\BloodPressureEstimation\.venv\Lib\site-packages")
import gradio as gr
import glob
import os

def glob_expl(data_files):
    path = r"D:\minowa\BloodPressureEstimation\data\raw\ppgabp\*"
    return gr.FileExplorer(glob=path,height=250)
    # return None

default_path = r"D:\data\*"
data_files = gr.FileExplorer(glob=default_path,height=300)
demo = gr.Interface(
    glob_expl,
    data_files,
    data_files,
    # live=True
)
    # with gr.Row():
    #     data_button = gr.Button("Update")
        
    # data_button.click(glob_expl,outputs=data_files)


if __name__ == "__main__":
    demo.launch()