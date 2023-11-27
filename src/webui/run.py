import sys
sys.path.append(r"F:\minowa\BloodPressureEstimation\.venv\Lib\site-packages")
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from plot import InteractivePlot

plot_generator = InteractivePlot()
default_path = r"F:\minowa\BloodPressureEstimation\data\raw\ppgabp"
default_path_palette = r"F:\minowa\BloodPressureEstimation\repos\Pallette\experiments"
with gr.Blocks() as demo:
    with gr.Tab("Signal Preprocessing"):
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
            slider_zoom = gr.Slider(minimum = 1, maximum = 3,step=0.5,label="Zoom",scale=5)
        
        plot = gr.Plot()
        plot2 = gr.Plot()
        slider_prev.click(lambda x,y:plot_generator.prev(x,y),inputs=[plot,plot2],outputs=[plot,plot2])
        slider_next.click(lambda x,y:plot_generator.next(x,y),inputs=[plot,plot2],outputs=[plot,plot2])
    with gr.Tab("Synthesized signal"):
        with gr.Row():
            data_text_1 = gr.Textbox(value=default_path,max_lines=1,interactive=True,scale=5,label="Path to signal directory")
            data_button_1 = gr.Button("Update",size="sm",scale=1)
            data_dropdown_1 = gr.Dropdown(scale=6,label="Select file")
        with gr.Row():
            slider_prev_1 = gr.Button("◀",min_width=80,scale=1)
            slider_next_1 = gr.Button("▶",min_width=80,scale=1)
            # ae mse etc.
                
        with gr.Row():
            plot10 = gr.Plot("Condition")
            plot11 = gr.Plot("Ground Truth")
            plot12 = gr.Plot("Output")
            
    @data_button.click(inputs=data_text,outputs=data_dropdown_dir)
    def glob_p0n(path):
        l = glob.glob(path+r"\???")
        ret = [os.path.basename(f) for f in l]
        return gr.Dropdown(choices=ret,scale=1,label="Patients")
    
    @data_button_1.click(inputs=data_text_1,putputs=data_dropdown_1)
    @data_dropdown_dir.select(inputs=[data_text,data_dropdown_dir],outputs=data_dropdown_file)
    def glob_npy(data_path,data_dirs):
        if data_dirs is None:
            root = data_path
        else:
            root =os.path.join(data_path,data_dirs)
        ret = glob.glob("*\*.*",root_dir=root)
        return gr.Dropdown(choices=ret,interactive=True,scale=6,label="Select file")

    @data_dropdown_1.select(inputs=[data_text_1,data_dropdown_1],outputs=[plot10,plot11,plot12])
    def update_tab2():
    
    @data_dropdown_file.select(inputs=[data_text,data_dropdown_dir,data_dropdown_file,plot,plot2],outputs=[slider_scroll,slider_zoom, plot,plot2])
    def update_plot(dir1,dir2,dir3,plot,plot2):
        if plot != None or plot2 != None:
            plt.clf()
            plt.close()
            
        path = os.path.join(dir1,dir2,dir3)
        sig = np.load(path)
        plot_generator.set_signal(sig)
        # print(sig.dtype,sig.shape)
        # print(sig[:10])
        
        return gr.Slider(minimum=0,maximum =len(sig), label="Scroll",scale=7),gr.Slider(minimum=1,maximum=int(np.log10(len(sig))+1),step=0.5,label="Zoom",scale=3),*plot_generator.plot()
    
    @slider_scroll.release(inputs=[slider_scroll,plot,plot2],outputs=[plot,plot2])
    def scroll_plot(t,plot,plot2):
        # print(plot.keys())
        p1,p2=plot_generator.set_scroll([plot,plot2],t)
        return p1,p2
    @slider_zoom.release(inputs=[slider_zoom,plot,plot2],outputs=[plot,plot2])
    def zoom_plot(x,plot,plot2):
        p1,p2= plot_generator.set_zoom([plot,plot2],x)
        return p1,p2
    # button = gr.Radio(label="Plot type",
    #                   choices=['scatter_plot', 'heatmap', 'us_map',
    #                            'interactive_barplot', "radial", "multiline"], value='scatter_plot')
    # plot = gr.Plot(label="Plot")
    # button.change(make_plot, inputs=[button,plot], outputs=[plot])
    # demo.load(make_plot, inputs=[button], outputs=[plot])

if __name__ == "__main__":
    demo.launch()