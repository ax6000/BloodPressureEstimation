import numpy as np
import matplotlib.pyplot as plt
from scipy import signal,stats
import logging

def butterworth(sig,N=4,btype='bandpass',Wn=[0.5,8],fs=125):
    sos = signal.butter(N,Wn,btype,fs=fs,output='sos')
    return signal.sosfiltfilt(sos,sig)

def hampel(sig,k=3,nsigma=3):
    median = signal.medfilt(sig,kernel_size=k)
    deviation = np.abs(sig-median)
    median_deviation = signal.medfilt(deviation,kernel_size=k)
    threshold = 1.4826 * nsigma*median_deviation
    outliers = np.where(deviation > threshold)[0]
    sig[outliers] = median[outliers]
    return sig
    
def minmax_norm(sig):
    return (sig-np.nanmin(sig))/(np.nanmax(sig)-np.nanmin(sig))
# def zscore_norm(sig):
    
def detect_flat(sig,window,t_flat):
    sig = sig.T
    len_data = sig.shape[1]
    flat_locs_abp = np.ones((len_data - window+1,), dtype=int)
    flat_locs_ppg = np.ones((len_data - window+1,), dtype=int)
    # print(flat_locs_abp.shape)
    # Get the locations where i == i+1 == i+2 ... == i+window
    # efficient-ish sliding window
    # print(data[1, :len_data - window].shape,data[1, 0:len_data - window + 0].shape)
    for i in range(1, window):
        # print(data[1, :len_data - window+1].shape, data[1, i:len_data - window + i+1].shape)
        tmp_abp = (sig[1, :len_data - window+1] == sig[1, i:len_data - window + i+1])
        tmp_ppg = (sig[0, :len_data - window+1] == sig[0, i:len_data - window + i+1])
        flat_locs_abp = (flat_locs_abp & tmp_abp)
        flat_locs_ppg = (flat_locs_ppg & tmp_ppg)
        # print(tmp_abp.shape,tmp_ppg.shape)
        # Extend to be the same size as data
    flat_locs_ppg = np.concatenate((flat_locs_ppg, np.zeros((window - 1,), dtype=bool)))
    flat_locs_abp = np.concatenate((flat_locs_abp, np.zeros((window - 1,), dtype=bool)))
    # print(flat_locs_abp.shape,flat_locs_ppg.shape)
    flat_locs_ppg2 = flat_locs_ppg.copy()
    flat_locs_abp2 = flat_locs_abp.copy()
    # print(flat_locs_abp2.shape,flat_locs_ppg2.shape)
        
    # Mark the ends of the window
    for i in range(1, window):
        flat_locs_abp[i:] = flat_locs_abp[i:] | flat_locs_abp2[:len_data-i]
        flat_locs_ppg[i:] = flat_locs_ppg[i:] | flat_locs_ppg2[:len_data-i]
        
    # Percentages
    per_abp = np.sum(flat_locs_abp) / len_data
    per_ppg = np.sum(flat_locs_ppg) / len_data
    if per_abp > t_flat or per_ppg > t_flat:
        logging.info(f"invalid because of flat lines: {per_ppg}, {per_abp},{np.sum(flat_locs_abp)},{np.sum(flat_locs_ppg)},{np.nanmax(flat_locs_abp)},{np.nanmax(flat_locs_ppg)}")
        return True,None,None
    else:
        return False,flat_locs_ppg,flat_locs_abp
    
def flat(sig,plot=False,xfrom=0,xlen=100000):
    if sig is None:
        return None
    w_flat = 15    # flat lines window
    w_peaks = 5    # flat peaks window
    w_fix = 15     # flat join window
    t_nan = 125 * 120
    # thresholds
    t_peaks = 0.05 # percentage of tolerated flat peaks
    t_flat = 0.1  # percentage of tolerated flat lines
    if plot:
        sig_processed = sig.copy()
    # 0 nanでorをとる
    nan_mask = np.isnan(sig[:,0]) | np.isnan(sig[:,1])
    s = np.sum(~nan_mask)
    if  s < t_nan:
        logging.info(f"invalid because of short valid signal: {s}")
        return None
    sig[nan_mask] = np.nan
    
    # 1, 2
    is_invalid,flat_locs_ppg,flat_locs_abp=detect_flat(sig,w_flat,t_flat)
    if is_invalid:
        return None
    # 3 find peaks, calcurate flat peaks rate
    peaks_ppg, peaks_info_ppg = signal.find_peaks(sig[:,0],distance=35,plateau_size=1)
    peaks_abp, peaks_info_abp = signal.find_peaks(sig[:,1],distance=35,plateau_size=1)
    # 4 delete data if too much flat lines/peaks
    if len(peaks_info_ppg['plateau_sizes']) == 0 or len(peaks_info_abp['plateau_sizes']) == 0:
        return None
    len_data=len(sig)
    per_abp =len(np.where(peaks_info_ppg['plateau_sizes']>w_peaks)[0])/ len(peaks_info_ppg['plateau_sizes'])
    per_ppg =len(np.where(peaks_info_abp['plateau_sizes']>w_peaks)[0])/len(peaks_info_abp['plateau_sizes'])
    # del
    if per_abp > t_peaks or per_ppg > t_peaks:
        logging.info(f"invalid because of flat lines: {per_ppg}, {per_abp}")
        return None
    # 5 find valleys
    valleys_ppg, _ = signal.find_peaks(-sig[:,0],distance=35)
    valleys_abp, _ = signal.find_peaks(-sig[:,1],distance=35)
    # 6 fill valley-valley with nan if flat line exists
    # connect_valley_locs_???[i] = True if there's flat line between valleys[i] and valleys[i+1] 
    # ppg
    connect_valley_locs_ppg = np.zeros_like(valleys_ppg)
    for i in range(valleys_ppg.shape[0]-1):
        connect_valley_locs_ppg[i] = np.any(flat_locs_ppg[valleys_ppg[i]:valleys_ppg[i+1]])
    # print("total connecting valleys:",np.sum(connect_valley_locs_ppg),valleys_ppg.shape[0]-1)
    for j in range(connect_valley_locs_ppg.shape[0]-1):
        if connect_valley_locs_ppg[j]:
            sig[valleys_ppg[j]:valleys_ppg[j+1],:]=np.nan 
    # abp
    connect_valley_locs_abp = np.zeros_like(valleys_abp)
    for i in range(valleys_abp.shape[0]-1):
        connect_valley_locs_abp[i] = np.any(flat_locs_abp[valleys_abp[i]:valleys_abp[i+1]])
    for j in range(connect_valley_locs_abp.shape[0]-1):
        if connect_valley_locs_abp[j]:
            sig[valleys_abp[j]:valleys_abp[j+1],:]=np.nan 
    sig[np.where(flat_locs_ppg)[0],:] = np.nan
    sig[np.where(flat_locs_abp)[0],:] = np.nan
    # 7 nan del
    starts_with_nan = np.isnan(sig[0,0])
    nan_diff = np.diff(np.isnan(sig[:,0]))
    nan_border = np.where(nan_diff==1)[0]
    if not starts_with_nan:
        nan_border = nan_border[1:-1]
    if len(nan_border) % 2 == 1:
        nan_border = nan_border[:-1]
    nan_border = nan_border.reshape(-1,2)
    nan_length = nan_border[:,1]-nan_border[:,0]
    
    # print(nan_border[:5],nan_length[:30])
    for i,l in enumerate(nan_length):
        if l < t_nan:
            sig[nan_border[i,0]:nan_border[i,1],:]=np.nan
    # print(len(nan_diff_pos),nan_diff_pos[:5],len(nan_diff_neg),nan_diff_neg[:5])
    nan_mask = np.isnan(sig[:,0])
    # leave single nan
    nan_border = nan_border[np.where(nan_length < t_nan)[0]]
    nan_mask[nan_border] = False
    sig = sig[~nan_mask,:]
    if len(sig) < t_nan:
        return None
    # print("before:",len_data,"after:",len(sig))
    if plot:
        x = np.arange(0,len(sig))
        plt.figure(figsize=(12,4))
        plt.subplot(2,1,1)
        # plt.plot(x,sig_processed[:,0],label='-.',color='black')
        plt.plot(x,sig[:,0])
        # plt.scatter(x[valleys_ppg], sig[valleys_ppg,0], color='red',s=24)
        plt.xlim(xfrom,xfrom+xlen)
        plt.subplot(2,1,2)
        # plt.plot(x,sig_processed[:,1],label='-.',color='black')
        plt.plot(x,sig[:,1])
        # plt.scatter(x[valleys_abp], sig[valleys_abp,1], color='red',s=24)
        plt.xlim(xfrom,xfrom+xlen)
        plt.show()
    return sig
