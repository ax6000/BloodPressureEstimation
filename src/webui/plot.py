import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy.signal import find_peaks

class InteractivePlot():
    
    def __init__(self) -> None:
        self.sig  =None
        self.length = None
        self.length_log10 = None
        self.xstart = 0
        self.xend = self.length
        self.zoom = 1
        self.window = self.length
        self.FIGSIZE=(12,6)
        self.ppg_lim = []
        self.abp_lim = []
        
    def set_signal(self,sig):
        self.sig = sig
        self.length = len(sig)
        self.xstart = 0
        self.xend = self.length
        self.length_log10 = np.log10(self.length)
        self.range = np.arange(0,self.length)
        self.abp_lim = [np.nanmin(sig[:,0])-5,np.nanmax(sig[:,0])+5]
        self.ppg_lim = [np.nanmin(sig[:,1])-0.1,np.nanmax(sig[:,1])+0.1]
        self.calc_peaks()
        if self.xend is None:
            self.xend = self.length
        if self.window is None:
            self.window = self.length
    
    
    def calc_peaks(self,w_peaks=5):
        self.ppg_peaks, pinfo_ppg = find_peaks(self.sig[:,0],plateau_size=1)
        self.abp_peaks, pinfo_abp = find_peaks(self.sig[:,1],plateau_size=1)
        self.ppg_plateau = self.ppg_peaks[np.where(pinfo_ppg['plateau_sizes']>w_peaks)[0]]
        self.abp_plateau = self.abp_peaks[np.where(pinfo_abp['plateau_sizes']>w_peaks)[0]]
        print(self.ppg_peaks.shape,self.abp_peaks.shape)
        
    def set_scroll(self,plots,t):
        self.xstart = t
        self.xend = min(t+self.window,self.length)
        self.window = max(self.window,int(10**self.zoom) )
        self.window = min(self.window,self.length-self.xstart)
        return self.plot(plots)
    
    def set_zoom(self,plots,x):
        self.zoom = x
        self.window = min(int(10**x),self.length-self.xstart)
        self.xend = min(self.xstart+self.window,self.length)
        return self.plot(plots)
    
    def next(self,p1,p2):
        return self.set_scroll([p1,p2],min(self.length-1,self.xstart+self.window))
    
    def prev(self,p1,p2):
        return self.set_scroll([p1,p2],max(0,self.xstart-self.window))
        

    def plot_normal(self):
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
        axes[0].set_ylim(self.abp_lim)
        axes[1].set_ylim(self.ppg_lim)
        fig.suptitle(f"{self.xstart}-{self.xend}")
        fig.tight_layout()
        return fig
    
    def plot_peaks(self,w_peaks=5):
        fig,axes = plt.subplots(2,1,figsize=self.FIGSIZE)
        p_ppg = self.ppg_peaks[(self.xstart<=self.ppg_peaks) & (self.ppg_peaks<self.xend)]
        pl_ppg = self.ppg_plateau[(self.xstart<=self.ppg_plateau) & (self.ppg_plateau<self.xend)]
        p_abp = self.abp_peaks[(self.xstart<=self.abp_peaks) & (self.abp_peaks<self.xend)]
        pl_abp = self.abp_plateau[(self.xstart<=self.abp_plateau) & (self.abp_plateau<self.xend)]
        axes[0].plot(self.range[self.xstart:self.xend],self.sig[self.xstart:self.xend, 0], 'black',linewidth=1)
        axes[0].scatter(self.range[p_ppg], self.sig[p_ppg,0], color='red',s=24)
        axes[0].scatter(self.range[pl_ppg], self.sig[pl_ppg,0], color='blue',s=24)

        axes[1].plot(self.range[self.xstart:self.xend],self.sig[self.xstart:self.xend, 1], 'black',linewidth=1)
        axes[1].scatter(self.range[p_abp], self.sig[p_abp, 1], color='red',s=24)
        axes[1].scatter(self.range[pl_abp], self.sig[pl_abp, 1], color='blue',s=24)
        
        axes[0].set_ylabel("ABP/mmHg")
        axes[1].set_ylabel("PPG/PU")
        # tick_digit = int(np.log10(self.window))
        # mult = 10**tick_digit
        # tick = np.arange(0,self.window,mult//5)
        # axes[0].set_xticks(tick)
        # axes[1].set_xticks(tick)
        axes[0].set_ylim(self.abp_lim)
        axes[1].set_ylim(self.ppg_lim)
        fig.suptitle(f"{self.xstart}-{self.xend}")
        fig.tight_layout()
        return fig
    
    def plot(self,plots=None):
        if plots != None:
            plt.clf()
            plt.close()
        # print(self.xstart,self.xend)
        return self.plot_normal(),self.plot_peaks()
    