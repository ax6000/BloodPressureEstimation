import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

class InteractivePlot():
    def __init__(self) -> None:
        self.sig  =None
        self.length = None
        self.length_log10 = None
        self.xstart = 0
        self.xend = self.length
        self.zoom = 1
        self.window = self.length
        self.FIGSIZE=(10,5)
    def set_signal(self,sig):
        self.sig = sig
        self.length = len(sig)
        self.length_log10 = np.log10(self.length)
        if self.xend is None:
            self.xend = self.length
        if self.window is None:
            self.window = self.length
    def set_scroll(self,plot,t):
        self.xstart = t
        self.xend = min(t+self.window,self.length)
        self.window = max(self.window,10**self.zoom )
        self.window = min(self.window,self.length-self.xstart)
        return self.plot(plot)
    def set_zoom(self,plot,x):
        self.zoom = x
        self.window = min(10**x,self.length-self.xstart)
        self.xend = min(self.xstart+self.window,self.length)
        return self.plot(plot)
    def next(self,plot):
        return self.set_scroll(plot,min(self.length-1,self.xstart+self.window))
    def prev(self,plot):
        return self.set_scroll(plot,max(0,self.xstart-self.window))
        
    def plot(self,plot=None):
        if plot != None:
            plt.close(plot['plot'])
        # print(self.xstart,self.xend)
        fig,axes = plt.subplots(2,1,figsize=self.FIGSIZE)
        axes[0].plot(self.sig[self.xstart:self.xend,0])
        axes[1].plot(self.sig[self.xstart:self.xend,1])
        axes[0].set_ylabel("ABP/mmHg")
        axes[1].set_ylabel("PPG/PU")
        tick_digit = int(np.log10(self.window))
        mult = 10**tick_digit
        tick = np.arange(0,self.window,mult//5)
        axes[0].set_xticks(tick)
        axes[1].set_xticks(tick)
        axes[0].set_ylim([40,200])
        axes[1].set_ylim([0,1])
        fig.suptitle(f"{self.xstart}-{self.xend}")
        fig.tight_layout()
        return fig