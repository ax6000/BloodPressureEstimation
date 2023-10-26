import sys
sys.path.append(r"D:\minowa\BloodPressureEstimation\.venv\Lib\site-packages")
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from plot import InteractivePlot

plot_generator = InteractivePlot()
default_path = r"D:\minowa\BloodPressureEstimation\data\raw\ppgabp"
with gr.Blocks() as demo:
    with gr.Row():
        data_text = gr.Textbox(value=default_path,max_lines=1,interactive=True,scale=5,label="Path to signal directory")
        data_button = gr.Button("Update",size="sm",scale=1)
        data_dropdown_dir = gr.Dropdown(glob.glob("???",root_dir=default_path+"\\"),scale=1,label="Patients")
        data_dropdown_file = gr.Dropdown(scale=6,label="Select file")
        # @data_button.click(inputs=data_text,outputs=[data_dropdown,data_expl])
    with gr.Row():
        slider_scroll = gr.Slider(label="Scroll",scale=13)
        slider_prev = gr.Button("◀",min_width=80,scale=1)
        slider_next = gr.Button("▶",min_width=80,scale=1)
        slider_zoom = gr.Slider(minimum = 1, maximum = 3,step=1,label="Zoom",scale=5)
    
    plot = gr.Plot()
    slider_prev.click(lambda x:plot_generator.prev(x),inputs=plot,outputs=plot)
    slider_next.click(lambda x:plot_generator.next(x),inputs=plot,outputs=plot)
    
    @data_button.click(inputs=data_text,outputs=data_dropdown_dir)
    def glob_p0n(path):
        l = glob.glob(path+r"\???")
        ret = [os.path.basename(f) for f in l]
        return gr.Dropdown(choices=ret,scale=1,label="Patients")
        
    @data_dropdown_dir.select(inputs=[data_text,data_dropdown_dir],outputs=data_dropdown_file)
    def glob_npy(data_path,data_dirs):
        if data_dirs is None:
            root = data_path
        else:
            root =os.path.join(data_path,data_dirs)
        ret = glob.glob("*\*.*",root_dir=root)
        return gr.Dropdown(choices=ret,interactive=True,scale=6,label="Select file")

    @data_dropdown_file.select(inputs=[data_text,data_dropdown_dir,data_dropdown_file,plot],outputs=[slider_scroll,slider_zoom, plot])
    def update_plot(dir1,dir2,dir3,plot):
        if plot != None:
            plt.close(plot['plot'])
        path = os.path.join(dir1,dir2,dir3)
        sig = np.load(path)
        plot_generator.set_signal(sig)
        # print(sig.dtype,sig.shape)
        # print(sig[:10])
        
        return gr.Slider(minimum=0,maximum =len(sig), label="Scroll",scale=7),gr.Slider(minimum=1,maximum=int(np.log10(len(sig))+1),step=1,label="Zoom",scale=3),plot_generator.plot()
    
    @slider_scroll.release(inputs=[slider_scroll,plot],outputs=plot)
    def scroll_plot(t,plot):
        # print(plot.keys())
        return plot_generator.set_scroll(plot,t)
    @slider_zoom.release(inputs=[slider_zoom,plot],outputs=plot)
    def zoom_plot(x,plot):
        return plot_generator.set_zoom(plot,x)
        
    # button = gr.Radio(label="Plot type",
    #                   choices=['scatter_plot', 'heatmap', 'us_map',
    #                            'interactive_barplot', "radial", "multiline"], value='scatter_plot')
    # plot = gr.Plot(label="Plot")
    # button.change(make_plot, inputs=[button,plot], outputs=[plot])
    # demo.load(make_plot, inputs=[button], outputs=[plot])

if __name__ == "__main__":
    demo.launch()