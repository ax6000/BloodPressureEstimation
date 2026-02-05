'''
This module includes functions and classes to extract different features from PPG signals. The module is able to extract two kind of features: event features, such as peaks or cycles, and statisticals features for future data analysis. The main functions and classes are:

'''

import numpy as np
import pyampd
import scipy.signal
import copy

# from .sqi import kurtosis, skew
# from .preprocessing import waveform_norm, mean_filter_normalize


#-------- General functions -------# 


def _compute_cyle_pks_vlys(sig, fs, pk_th=0.6, remove_start_end = True):
    """
    Extract the peaks and valleys that delimits the cardiac cycles of the signal waveform passed parameter.
    Peaks with an amplitude under 'pk_th' of previous peak are considered diastolic peaks and ignored.

    Parameters
    ----------
    sig: array
        Signal waveform
    fs: int
        Frequency sampling rate (Hz)
    pk_th: float
        Threshold to identify diastolic peaks (0.6 by default)  
    remove_start_end: bool
        Enable to remove first and last peak or valley found.
    
    Returns
    -------
    bool
        Flag indicating if there are peaks identified as diastolic peak (True).
    bool
        Flag indicating if signal does not follow peak-valley-peak structure (True).
    array
        Indices of the peaks of the signal waveform.
    array
        Indices of the valleys of the signal waveform.
    """

    peaks = pyampd.ampd.find_peaks(sig, scale=int(fs))
    valleys = pyampd.ampd.find_peaks(sig.max()-sig, scale=int(fs))

    flag1, flag2 = False, False
    
    ### Remove first or last if equal to 0 or len(sig)-1
    if peaks[0] == 0: peaks = peaks[1:]
    # if valleys[0] == 0: valleys = valleys[1:]
    if peaks[-1] == len(sig)-1: peaks = peaks[:-1]
    if valleys[-1] == len(sig)-1: valleys = valleys[:-1]

    ### HERE WE SHOULD REMOVE THE FIRST AND LAST PEAK/VALLEY
    if remove_start_end:
        if peaks[0] < valleys[0]: peaks = peaks[1:]
        else: valleys = valleys[1:]

        if peaks[-1] > valleys[-1]: peaks = peaks[:-1]
        else: valleys = valleys[:-1]
    
    ### START AND END IN VALLEYS
    while len(peaks)!=0 and peaks[0] < valleys[0]:
        peaks = peaks[1:]
    
    while len(peaks)!=0 and peaks[-1] > valleys[-1]:
        peaks = peaks[:-1]
    
    if len(peaks)==0 or len(valleys)==0:
        return True, True, [], []
        
    ## Remove consecutive peaks with one considerably under the other
    new_peaks = []
    mean_vly_amp = np.mean(sig[valleys])
    # define base case:
    
    for i in range(len(peaks)-1):
        if sig[peaks[i]]-mean_vly_amp > (sig[peaks[i+1]]-mean_vly_amp)*pk_th:
            new_peaks.append(peaks[i])
            a=i
            break
            
    if len(peaks) == 1:
        new_peaks.append(peaks[0])
        a=0

    for j in range(a+1,len(peaks)):
        if sig[peaks[j]]-mean_vly_amp > (sig[new_peaks[-1]]-mean_vly_amp)*pk_th:
            new_peaks.append(peaks[j])
            
    if not np.array_equal(peaks,new_peaks):
        flag1 = True
        
    if len(valleys)-1 != len(new_peaks):
        flag2 = True
        
    if len(valleys)-1 == len(new_peaks):
        for i in range(len(valleys)-1):
            if not(valleys[i] < new_peaks[i] and new_peaks[i] < valleys[i+1]):
                flag2 = True
        
    return flag1, flag2, new_peaks, valleys


def compute_sp_dp(sig, fs, pk_th=0.6, remove_start_end = False):
    """
    Compute SBP and DBP as the median of the amplitude of the systolic peaks and diastolic valleys.
    The extracted peaks and valleys that delimits the cardiac cycles are extracted from the signal waveform passed parameter.
    Peaks whose amplitude is under 'pk_th' of previous peak are considered diastolic peaks and ignored.

    Parameters
    ----------
    sig : array
        Signal waveform
    fs: int
        Frequency sampling rate (Hz)
    pk_th: float
        Threshold to identify diastolic peaks (0.6 by default)  
    remove_start_end: bool
        Enable to remove first and last peak or valley found.
    
    Returns
    -------
    bool
        Flag indicating if there are peaks identified as diastolic peak (True).
    bool
        Flag indicating if signal does not follow peak-valley-peak structure (True).
    array
        Indices of the peaks of the signal waveform.
    array
        Indices of the valleys of the signal waveform.
    """

    flag1, flag2, new_peaks, valleys = _compute_cyle_pks_vlys(sig, fs, pk_th=pk_th, remove_start_end = remove_start_end)

    if len(new_peaks)!=0 and len(valleys)!=0:
        sp, dp = np.median(sig[new_peaks]), np.median(sig[valleys])
    else:
        sp, dp = -1 , -1

    return sp, dp, flag1, flag2, new_peaks, valleys


def extract_cycle_check(sig, fs, pk_th=0.6, remove_start_end = True):
    """
    Extract the cardiac cycles of the signal waveform passed parameter.
    Peaks whose amplitude is under 'pk_th' of previous peak are considered diastolic peaks and ignored.

    Parameters
    ----------
    sig : array
        Signal waveform
    fs: int
        Frequency sampling rate (Hz)
    pk_th: float
        Threshold to identify diastolic peaks (0.6 by default)  
    remove_start_end: bool
        Enable to remove first and last peak or valley found.
    
    Returns
    -------
    array
        cycles of the signal. 
    array
        Indices of the systolic peaks of each cycle (normalized to each cycle length)
    bool
        Flag indicating that a peak was identified as diastolic peak (if True).
    bool
        Flag indicating that signal does not follow peak-valley-peak structure (if True).
    array
        Indices of the peaks of the signal waveform.
    array
        Indices of the valleys of the signal waveform.
    """

    flag1, flag2, new_peaks, valleys = _compute_cyle_pks_vlys(sig, fs, pk_th=pk_th, remove_start_end = remove_start_end)

    cycles = []
    peaks_norm = []

    if len(new_peaks)!=0 and len(valleys) !=0:
        ## Save segments
        for i in range(len(valleys)-1):
            #print((valleys[i],valleys[i+1]))
            cycles.append(sig[valleys[i]:valleys[i+1]])
            
        ## Save peaks
        if len(valleys)-1 == len(new_peaks):
            for i in range(len(new_peaks)):
                peaks_norm.append(new_peaks[i]-valleys[i])
    
    return cycles, peaks_norm, flag1, flag2, new_peaks, valleys


def extract_feat_cycle(cycles, peaks_norm, fs,mean=True,ver2=False):
    """
    Extracts the time-based features of each cycle and ouputs their average.

    Parameters
    ----------
    cycles : array
        Cycles of a PPG signal waveform. 
    peaks_norm : array
        Indices of the systolic peaks of each cycle (normalized to each cycle length)
    fs: int
        Frequency sampling rate (Hz)

    Returns
    -------
    array
        Header with a list of the name of each feature. 
    array
        Average of the cycles' features
    """
    
    feats = []
    feat_name = []

    for c, p in zip(cycles, peaks_norm):
        # try:
        feat_name,feat= extract_temp_feat(c, p, fs,ver2=ver2)
        # append a zero to the end of the feature name
        feat = np.append(feat, c[0])  # append a zero to the end of the feature
        # nan_cols = ['apg_a', 'apg_b', 'apg_c', 'apg_d', 'apg_e']
        nan_idx = [55, 56, 57, 58, 59]
        # raise Exception(str(nan_idx))
        # t_cols = ['T_b', 'T_c', 'T_d', 'T_e']
        t_idx = [74, 75, 76, 77]
        # raise Exception(str(t_idx))
        n_abcde = (feat[t_idx] < 0).any()
        nan_abcde =  np.isnan(feat[nan_idx]).any()
        feat = np.append(feat, n_abcde)  # append number of cycles
        feat = np.append(feat, nan_abcde)  # append number of cycles
        feats.append(feat)
        
        feat_name = np.append(feat_name, 'cycle_zero')
        feat_name = np.append(feat_name, 'n_abcde')
        feat_name = np.append(feat_name, 'n_nan_cycle')  # append cycle length to the feature name
        
        # except Exception as e:
        #     print("Cycle ignored;",e)
    
    if len(feats)>0:
        if mean:
            feats_tmp = np.vstack(feats)
            feats = np.nanmean(feats_tmp, axis=0)
            feats[-3:] = np.nansum(feats_tmp[:,-3:], axis=0)  # sum n_cycle and n_abcde
            feat_name= np.append(feat_name, 'n_cycle')
            feats = np.append(feats, len(feats_tmp))  # append number of cycles
        else:
            feats = np.vstack(feats)

    else:
        feats = np.array([])
        
    return feat_name, feats



def extract_feat_original(sig, fs, filtered=True, remove_start_end=True):
    """
    Extracts other set of features from PPG signal waveform

    Parameters
    ----------
    sig : array
        PPG signal waveform. 
    fs: int
        Frequency sampling rate (Hz)
    filtered : array
        Filter derivatives to compute the features.
    remove_start_end : array
        Enable to remove first and last peak/valley in cycle identification.

    
    Returns
    -------
    array
        Header with a list of the name of each feature. 
    array
        Extracted features.
    """

    ppg = PPG(sig,fs)
    # print("calling features_extractor")
    _, head, feat_str = ppg.features_extractor(filtered=filtered, remove_first=remove_start_end)
    
    feat = [float(s) for s in feat_str.split(', ')]
    
    return head, feat


#-------- Cycle based temporal features -------# 

def width_at_per(per, cycle, peak, fs):
    """
    Extract the width of the systolic and diastolic phases at ('per'*100)% of amplitude of systolic peak

    Parameters
    ----------
    per: float
        ratio of the amplitude of the systolic peak to compute the width.
    cycle: array
        PPG cycle waveform. 
    peak: int
        Index of the systolic peak.
    fs: int
        Frequency sampling rate (Hz)

    Returns
    -------
    float
        Width of the systolic phase at ('per'*100)%  of amplitude of systolic peak.
    float
        Width of the diastolic phase at ('per'*100)%  of amplitude of systolic peak.
    """
    
    height_to_reach = cycle[peak]*per
    
    i = 0 
    while i < peak and cycle[i] < height_to_reach:
        i+=1
    
    SW = peak - i
    
    i = peak
    while i < len(cycle) and cycle[i] > height_to_reach:
        i +=1
    i -= 1
    
    DW = i - peak
    
    return SW, DW 
    

def vpg_points(vpg, peak):
    """
    Extract VPG or FDPPG interest points:
        w: maximum value between start and peak, same as Steepest
        y: relevant valley after w (after peak), same as NegSteepest
        z: maximum value after w with limit search, peak after y, same as TdiaRise

    Parameters
    ----------
    vpg : array
        VPG or FDPPG signal waveform. 
    peak: int
        Index of the systolic peak.

    Returns
    -------
    int
        Location (index) in VPG of w.
    int
        Location (index) in VPG of y.
    int
        Location (index) in VPG of z.
    """
    
    pks = pyampd.ampd.find_peaks(vpg)
    vlys = pyampd.ampd.find_peaks(-vpg)
    
    #Time from cycle start to first peak in VPG (steepest point)
    #steepest point == max value between start and peak
    Tsteepest = np.argmax(vpg[:peak])
    w = Tsteepest 
    
    pks = pks[pks > peak]
    vlys = vlys[vlys > peak]

    if len(vlys) < 1:
        # min slope in ppg' and max slope in ppg'
        end = int((len(vpg)-peak)*0.4)+peak
        y = np.argmin(vpg[peak:end])+peak
    else:
        y = vlys[0]
        
    min_slope_idx = y

    #pks = pks[pks > y]
    
    # Find the max (diatolic rise) from the prev. min to the end/2
    end = int((len(vpg)-min_slope_idx)*0.4)+min_slope_idx
    #TdiaRise. Max positive slope after 'peak'
    TdiaRise = np.argmax(vpg[min_slope_idx:end])+min_slope_idx
    z = TdiaRise
    
    return w, y, z
   

def apg_points(apg, peak, w, y, z):
    """
    Extract APG or SDPPG interest points:
        a: point with the highest acceleration of the systolic upstroke (early systolic positive wave).
        b: deceleration point of the systolic phase (early systolic negative wave).
        c: first relevant peak in APG after the systolic peak (late systolic reincreasing wave).
        d: relevant and last valley of APG (late systolic redecreasing wave).
        e: dicrotic notch (early diastolic positive wave).

    Parameters
    ----------
    apg: array
        APG or SDPPG signal waveform. 
    peak: int
        Index of the systolic peak.
    w: int
        Index of the point w.
    y: int
        Index of the point y.
    z: int
        Index of the point z.

    Returns
    -------
    int
        Location (index) in APG of a.
    int
        Location (index) in APG of b.
    int
        Location (index) in APG of c.
    int
        Location (index) in APG of d.
    int
        Location (index) in APG of e.
    """
    a = b = c = d = e = 0
    
    #Limit the search to 60% of the cycle len
    pks = pyampd.ampd.find_peaks(apg[:int(len(apg)*0.6)])
    vly, _ = scipy.signal.find_peaks(-apg[:int(len(apg)*0.6)])
    
    if len(pks) < 1 or len(vly) < 1: 
        return a, b, c, d, e
    
    #### compute 'a' as the first peak of apg, if not max val before peak
  
    
    
    
    #a = np.argmax(apg[0:peak])
    a = pks[0]
    
    if a > peak:
        a = np.argmax(apg[0:peak])
    else:
        pks = pks[1:]
        
    vly = vly[vly > a]
    if len(vly) < 1: 
        return a, b, c, d, e
    
    # Reduce the valleys after w
    vly = vly[vly > w]
    if len(vly) < 1: 
        return a, b, c, d, e
    
    
    #### compute 'e' as the max peak after systolic peak 
    #pks_tmp = pks[pks > peak] # peaks after systolic peak
    pks_tmp = pks[pks > y] # peaks after y
    if len(pks_tmp)!=0:
        e = pks_tmp[np.argmax(apg[pks_tmp])]
    else: # max value after y
        e =  np.argmax(apg[y+1:])
    pks = pks[pks < e]
    vly = vly[vly < e]
    
    #### compute 'b' as the first valley of apg after a, if not min val before peak
    if len(vly[vly < peak-2]) < 1:
        vly_b, _ = scipy.signal.find_peaks(-apg[w:peak-2])
        vly_b+=w
        
        if len(vly_b) < 1:
            b = np.argmin(apg[w:peak-2])+w 
        else:
            b = vly_b[0]
    else:
        b = vly[0]
    
    
    #### compute 'd' as the min value after sys peak and e (or last valley)
    # Last valley between peak and e
    vly_tmp, _ = scipy.signal.find_peaks(-apg[peak:e])
    vly_tmp+=peak
    if len(vly_tmp) < 1:
        d = np.argmin(apg[peak:e])+peak
    else: # min valley after peak
        d = vly_tmp[np.argmin(apg[vly_tmp])]
        #d = vly_tmp[-1]
        
    #d = np.argmin(apg[peak:e])+peak
      
    #### compute 'c' as max peak between b and d or the max value between b and d
    pks_tmp, _ = scipy.signal.find_peaks(apg[b:d+1])
    pks_tmp+=b
    if len(pks_tmp) < 1:
        c = np.argmax(apg[b:d])+b
    else:
        c = pks_tmp[np.argmax(apg[pks_tmp])]
        #c = pks_tmp[-1]
    
    #c = np.argmax(apg[b:d])+b
    
    return a, b, c, d, e

import numpy as np
from scipy import signal
from scipy.signal import find_peaks

# -------------------------
# utility (classと同じ)
# -------------------------
def zerocrossing(x):
    """Find zero crossing points in signal x (class版と同じ)"""
    inew = 0
    r = x
    loc = []
    for ch in range(1, len(r)):
        if ((r[ch-1] < 0 and r[ch] > 0) or
            (r[ch-1] > 0 and r[ch] < 0)):
            loc.append(ch)
            inew += 1
    return np.array(loc, dtype=int)


# ============================================================
# 1) VPG points (PlotButtonPushedの該当部分をほぼコピー)
# ============================================================
def vpg_points2(
    vpg,
    peak,
    *,
    prominence_max=0.2/1000,
    prominence_min=0.2/1000,
    distance=7,
    require_two_maxima=True,
    return_debug=False,
):
    """
    Extract VPG or FDPPG interest points:
        w: maximum value between start and peak, same as Steepest
        y: relevant valley after w (after peak), same as NegSteepest
        z: maximum value after w with limit search, peak after y, same as TdiaRise

    ※元クラスの PlotButtonPushed 内のVPG peak検出〜 w,y,z割当を最大限維持

    Parameters
    ----------
    vpg : array
        VPG or FDPPG signal waveform.
    peak: int
        Index of the systolic peak. (ここでは保持するだけ。実際のロジックはクラス同様 vpg peaks 主導)
    prominence_max, prominence_min, distance: find_peaksのパラメータ
    require_two_maxima: Trueなら最大ピークが2個未満で例外
    return_debug: Trueならピーク配列も返す

    Returns
    -------
    w, y, z : int
        VPG上のインデックス
    (optional) debug dict
    """
    vpg = np.asarray(vpg)

    # Find peaks for VPG（クラスと同じ）
    VPG_peaks_max, _ = scipy.signal.find_peaks(vpg, prominence=prominence_max, distance=distance)
    VPG_peaks_min, _ = scipy.signal.find_peaks(-vpg, prominence=prominence_min, distance=distance)
    # Handle close points in APG（←コメントはAPGになってるけどクラス通り）
    if len(VPG_peaks_min) > 1 and len(VPG_peaks_max) > 0:
        if VPG_peaks_min[0] < VPG_peaks_max[0]:
            VPG_peaks_min = VPG_peaks_min[1:]

    if len(VPG_peaks_max) == 0:
        # if require_two_maxima:
        #     raise ValueError("Not enough VPG maxima points (0).")
        w = y = z = None
        if return_debug:
            return w, y, z, {"VPG_peaks_max": VPG_peaks_max, "VPG_peaks_min": VPG_peaks_min}
        return w, y, z

    vpg_max_idx = np.argmax(vpg[VPG_peaks_max])
    # print("VPG peaks max:", VPG_peaks_max,vpg_max_idx)
    if vpg_max_idx > 0:
        VPG_peaks_max = VPG_peaks_max[vpg_max_idx:]

    if len(VPG_peaks_max) < 2:
        # if require_two_maxima:
        #     raise ValueError(f"Not enough VPG maxima points: {VPG_peaks_max},{vpg_max_idx}")
        w = int(VPG_peaks_max[0])
        # y/zは可能な範囲で埋める（クラスはreturnして中断するが、関数なので返す）
        VPG_peaks_min = VPG_peaks_min[VPG_peaks_min >= VPG_peaks_max[0]]
        y = int(VPG_peaks_min[0]) if len(VPG_peaks_min) else None
        z = None
        if return_debug:
            return w, y, z, {"VPG_peaks_max": VPG_peaks_max, "VPG_peaks_min": VPG_peaks_min}
        return w, y, z

    VPG_peaks_min = VPG_peaks_min[VPG_peaks_min >= VPG_peaks_max[0]]

    # class: self.wEditField = VPG_peaks_max[0]
    #        self.yEditField = VPG_peaks_min[0]
    #        self.zEditField = VPG_peaks_max[1]
    w = int(VPG_peaks_max[0])
    y = int(VPG_peaks_min[0]) if len(VPG_peaks_min) else None
    z = int(VPG_peaks_max[1])

    if return_debug:
        return w, y, z, {"VPG_peaks_max": VPG_peaks_max, "VPG_peaks_min": VPG_peaks_min}
    return w, y, z

# ============================================================
# 呼び出し側の保管（クラス相当の最小例）
# ============================================================
def compute_T2_5_from_segment(min1, min2):
    # class: self.T2_5 = int(((min2 - min1 - 4) / 100) * 2.5)
    return int(((min2 - min1 - 4) / 100) * 2.5)


def compute_vpg_apg_jpg_from_seg(seg):
    """
    seg から class同様に VPG/APG/JPG を生成（平滑も同じ）
    """
    seg = np.asarray(seg)

    vpg = np.diff(seg) * 1000
    vpg = np.convolve(vpg, np.ones(6) / 6, mode='same')

    apg = np.diff(vpg) * 1000
    apg = np.convolve(apg, np.ones(8) / 8, mode='same')

    jpg = np.diff(apg) * 1000
    # class: rolling(window=85//8, center=True).mean()
    win = max(1, 85 // 8)
    # rolling meanの簡易版（端の扱いは完全一致しないが条件自体は同じように動く）
    jpg = np.convolve(jpg, np.ones(win) / win, mode="same")

    return vpg, apg, jpg





def apg_points2(
    apg,
    peak,
    w, y, z,
    *,
    jpg=None,
    T2_5=None,
    apg_maxima=None,
    apg_minima=None,
    apg_prominence=1.5,
    apg_distance=7,
    jpg_max_prominence=40,
    jpg_min_prominence=50,
    jpg_distance=6,
    spg_prominence=400,
    spg_distance=5,
    return_f_N_D=False,
    return_debug=False,
    log = False
):
    """
    元クラスの APG_c_d_test / cal_c_d / process_c_d_points をほぼそのまま維持しつつ、
    添字落ち・空配列・不足要素のときに落ちないように「ガードだけ」追加した安全版。

    返り値:
      (a,b,c,d,e) もしくは (a,b,c,d,e,f,N,D)
      計算不能な場合は (a,b,None,None,None, ...) のように None を混ぜて返す（落とさない）
    """
    apg = np.asarray(apg)

    def _has(arr, i):
        return arr is not None and len(arr) > i

    def _i(arr, i, default=None):
        return int(arr[i]) if _has(arr, i) else default

    def _clip_idx(idx, n):
        if idx is None:
            return None
        return int(np.clip(int(idx), 0, n - 1))

    # --- apg_maxima/minima が無ければ生成（ここは前回同様）
    if apg_maxima is None or apg_minima is None:
        apg_maxima, _ = scipy.signal.find_peaks(apg, prominence=apg_prominence, distance=apg_distance)
        apg_minima, _ = scipy.signal.find_peaks(-apg, prominence=apg_prominence, distance=apg_distance)

        if len(apg_maxima) > 0:
            apg_max_idx = np.argmax(apg[apg_maxima])
            if apg_max_idx > 0:
                apg_maxima = apg_maxima[apg_max_idx:]
            apg_minima = apg_minima[apg_minima >= apg_maxima[0]]
    # --- a,b は class通り先頭（不足ならNone）
    a = _i(apg_maxima, 0, None)
    b = _i(apg_minima, 0, None)
    if a >= b:
        a = b = None
    # --- jpg は必須（無ければ生成するが、T2_5 は class想定的に必須）
    if jpg is None:
        jpg = np.diff(apg)*1000
        win = max(1, 85 // 8)
        jpg = np.convolve(jpg, np.ones(win) / win, mode="same")
    jpg = np.asarray(jpg)

    if T2_5 is None:
        raise ValueError("T2_5 is required (same as class: computed in PlotButtonPushed).")

    # ---- 出力（c,d,e,f,N,D）
    c = d = e = f = N = D = None

    # =========================
    # process_c_d_points (安全化)
    # =========================
    def process_c_d_points(c_point, d_point, z_jpg):
        # z_jpg[2],[3] が無いと落ちるのでガード
        if z_jpg is None or len(z_jpg) < 4:
            return None  # classならこの後もいろいろ破綻するので中断扱いが近い

        cEditField = int(c_point)
        dEditField = int(d_point)
        eEditField = int(z_jpg[2])
        fEditField = int(z_jpg[3])
        NEditField = eEditField
        DEditField = fEditField
        return cEditField, dEditField, eEditField, fEditField, NEditField, DEditField

    # =========================
    # cal_c_d (安全化)
    # =========================
    def cal_c_d():
        # 必要最低限: apg_maxima[0] / apg_minima[0] が無いとフィルタ基準すら作れない
        if len(apg_maxima) == 0 or len(apg_minima) == 0:
            # print("error 712")
            return None

        # z_apg フィルタ（z_apg[1] を使う分岐があるので len>=2 か後で確認）
        z_apg = zerocrossing(apg)
        z_apg = z_apg[z_apg >= int(apg_maxima[0]) - 2]

        # JPG peaks/mins
        max_peaks, _ = scipy.signal.find_peaks(jpg, prominence=jpg_max_prominence, distance=jpg_distance)
        min_peaks, _ = scipy.signal.find_peaks(-jpg, prominence=jpg_min_prominence, distance=jpg_distance)

        # max_peaks = max_peaks[max_peaks >= int(apg_maxima[0]) - 2]
        # min_peaks = min_peaks[min_peaks >= int(apg_minima[0]) - 2]

        max_value_jpg = max_peaks[max_peaks >= int(apg_maxima[0]) - 2][:5]
        min_value_jpg = min_peaks[min_peaks >= int(apg_minima[0]) - 2]
   
        z_jpg = zerocrossing(jpg)
        z_jpg = z_jpg[z_jpg >= int(apg_maxima[0]) - 2]

        z_spg = zerocrossing(np.diff(jpg))
        # print("z_spg before limit:", z_spg, apg_maxima)
        z_spg = z_spg[z_spg >= int(apg_maxima[0]) - 2]

        if len(z_jpg) >= 3:
            z_spg = z_spg[z_spg <= z_jpg[2] + 2]
        z_spg = z_spg[z_spg <= apg_maxima[1]+2]
        if log:
            print("z_spg", z_spg,"z_jpg", z_jpg,"jpg pks", max_value_jpg[:2],"jpg vlys", min_value_jpg)
        # else: classは print するだけ（処理は続行）

        spg_peaks, _ = scipy.signal.find_peaks(np.diff(jpg), prominence=spg_prominence, distance=spg_distance)
        if len(spg_peaks) > 0 and spg_peaks[0] > apg_maxima[0]:
            spg_peaks = spg_peaks[1:]

        # ここから先は classの条件をそのまま使うが、
        # 参照する添字が存在するかだけを「その場で」チェックして足りなければ None を返す。

        # --- if len(min_value_jpg)>1 and jpg[min_value_jpg[1]] < 0
        if len(min_value_jpg) > 1 and jpg[min_value_jpg[1]] < 0:

            if len(z_spg) >= 3:
                # spg_peaks[1] が必要
                if len(spg_peaks) < 2:
                    if log:
                        print("error 2-a")
                    return None
                if log:
                    print("debug:case 2")
                return process_c_d_points(
                    int(z_spg[2] - T2_5),
                    int(z_spg[2] + T2_5),
                    z_jpg
                )

            elif len(z_jpg) >= 6 and (z is not None) and (z_jpg[3] < int(z)):
                if log:
                    print("debug:case 5")
                # z_jpg[2..5] が必要（len>=6は満たしてる）
                cEditField = int(z_jpg[2])
                dEditField = int(z_jpg[3])
                eEditField = int(z_jpg[4])
                fEditField = int(z_jpg[5])
                NEditField = eEditField
                DEditField = fEditField
                return cEditField, dEditField, eEditField, fEditField, NEditField, DEditField

            elif len(max_value_jpg) >= 1 and max_value_jpg[0] > min_value_jpg[0]:
                # z_apg[1] が必要
                if len(z_apg) < 2:
                    if log:
                        print("error 1-a")
                    return None
                if log:
                    print("debug:case 1-a")
                return process_c_d_points(int(max_value_jpg[0]), int(z_apg[1]), z_jpg)

            elif len(max_value_jpg) >= 2 and max_value_jpg[1] > min_value_jpg[0]:
                if len(z_apg) < 2:
                    if log:
                        print("error 1-ba")
                    return None
                if log:
                    print("debug:case 1-ba")
                return process_c_d_points(int(max_value_jpg[0]), int(z_apg[1]), z_jpg)

            elif len(max_value_jpg) >= 3 and max_value_jpg[2] > min_value_jpg[0]:
                if len(z_apg) < 2:
                    if log:
                        print("error 1-c")
                    return None
                if log:
                    print("debug:case 1-c")
                return process_c_d_points(int(max_value_jpg[1]), int(z_apg[1]), z_jpg)

            else:
                # Default: apg_maxima[1], apg_minima[1] が必要
                if len(apg_maxima) < 2 or len(apg_minima) < 2:
                    if log:
                        print("error default")
                    return None
                if log:
                    print("debug:case default")
                return process_c_d_points(int(apg_maxima[1]), int(apg_minima[1]), z_jpg)

        # --- elif len(min_value_jpg)>1 and jpg[min_value_jpg[1]] > 0
        elif len(min_value_jpg) > 1 and jpg[min_value_jpg[1]] > 0:

            if len(apg_maxima) > 1 and apg[int(apg_maxima[1])] < 0:
                # Case III: apg_maxima[1], apg_minima[1] が必要
                if len(apg_minima) < 2:
                    if log:
                        print("error 3")
                    return None
                if log:
                    print("debug:case 3")
                return process_c_d_points(int(apg_maxima[1]), int(apg_minima[1]), z_jpg)

            elif len(apg_maxima) > 1 and apg[int(apg_maxima[1])] >= 0:
                # Case II: spg_peaks[1] が必要
                if len(z_spg) < 3:
                    if log:
                        print("error 2+")
                    # print("error 809")
                    return None
                if log:
                    print("debug:case 2+")
                return process_c_d_points(
                    int(z_spg[2]- T2_5),
                    int(z_spg[2] + T2_5),
                    z_jpg
                )
            if log:
                print("error default 2")
            return None  # classには無いが、ここまで来たら何も返せないので安全にNone

        elif len(min_value_jpg) == 1:
            # print("error 820", max_peaks,min_peaks,apg_maxima,apg_minima,len(jpg))
            # print("error 820", "jpg std", np.std(jpg), "max",np.max(jpg), "min", np.min(jpg))
            if len(max_value_jpg) >= 1 and max_value_jpg[0] > min_value_jpg[0]:
                # z_apg[1] が必要
                if len(z_apg) < 2:
                    if log:
                        print("error 1-a-2")
                    return None
                if log:
                    print("debug:case 1-a-2")
                return process_c_d_points(int(max_value_jpg[0]), int(z_apg[1]), z_jpg)

            elif len(max_value_jpg) >= 2 and max_value_jpg[1] > min_value_jpg[0]:
                if len(z_apg) < 2:
                    if log:
                        print("error 1-ba-2")
                    return None
                if log:
                    print("debug:case 1-ba-2")
                return process_c_d_points(int(max_value_jpg[0]), int(z_apg[1]), z_jpg)

            elif len(max_value_jpg) >= 3 and max_value_jpg[2] > min_value_jpg[0]:
                if len(z_apg) < 2:
                    if log:
                        print("error 1-c-2")
                    return None
                if log:
                    print("debug:case 1-c-2")
                return process_c_d_points(int(max_value_jpg[1]), int(z_apg[1]), z_jpg)

            
            
            if log:
                print("error no jpg maxs")
            return None
        elif len(min_value_jpg) == 0:
            if log:
                print("error no jpg mins")
            return None
        else:
            # Default case: apg_maxima[1], apg_minima[1] が必要
            if len(apg_maxima) < 2 or len(apg_minima) < 2:
                if log:
                    print("error default2")
                return None
            if log:
                print("debug:case default2")
            return process_c_d_points(int(apg_maxima[1]), int(apg_minima[1]), z_jpg)

    # =========================
    # APG_c_d_test (安全化)
    # =========================
    if len(apg_maxima) > 1:

        # if apg[apg_maxima[1]] > 0:
        if apg[int(apg_maxima[1])] > 0:
            out = cal_c_d()
            if out is not None:
                c, d, e, f, N, D = out

        # elif apg[apg_maxima[1]] < 0 and (apg_maxima[1]-apg_minima[1]) < 20:
        elif (apg[int(apg_maxima[1])] < 0 and
              len(apg_minima) > 1 and  # ← ここだけガード（条件の意味は同じ、必要要素が無いなら判定不能）
              (int(apg_maxima[1]) - int(apg_minima[1])) < 20):
            if log:
                print("debug:case 3+")
            # class: maxima[2], minima[2] 参照があるのでガード
            c = int(apg_maxima[1])
            d = int(apg_minima[1])

            e = _i(apg_maxima, 2, None)
            f = _i(apg_minima, 2, None)
            N = e
            D = f

        else:
            out = cal_c_d()
            if out is not None:
                c, d, e, f, N, D = out

    else:
        out = cal_c_d()
        if out is not None:
            c, d, e, f, N, D = out

    # --- デバッグ情報
    debug = None
    if return_debug:
        debug = {
            "apg_maxima": apg_maxima,
            "apg_minima": apg_minima,
            "T2_5": T2_5,
            "w": w, "y": y, "z": z,
            "jpg_len": len(jpg),
            "peak": peak,
        }

    if return_f_N_D:
        if return_debug:
            return a, b, c, d, e, f, N, D, debug
        return a, b, c, d, e, f, N, D

    if return_debug:
        return a, b, c, d, e, debug
    return a, b, c, d, e


def extract_apg_feat(cycle, vpg, peak, w, y, z, fs,ver2=False):
    """
    Extract features related to interest points of APG (one cycle).

    Parameters
    ----------
    cycle: array
        PPG cycle waveform. 
    vpg: array
        VPG or FDPPG signal waveform. 
    peak: int
        Index of the systolic peak.
    w: int
        Index of the point w.
    y: int
        Index of the point y.
    z: int
        Index of the point z.
    fs: int
        Frequency sampling rate (Hz)

    Returns
    -------
    array
        Header with a list of the name of each feature. 
    array
        Extracted features related to interest points of APG (one cycle).
    """

    
    feats = []
    feats_header = []

    apg = np.diff(vpg)
    
    Tc = len(cycle)
    if ver2:
        apg2 = np.diff(vpg)*1000000
        apg2 = np.convolve(apg2, np.ones(6) / 6, mode='same')
        T2_5 = int(((len(cycle) - 4) / 100) * 2.5)
        a, b, c, d, e = apg_points2(apg2,peak,w, y, z,jpg=None,T2_5=T2_5,return_f_N_D=False)
    else:
        a, b, c, d, e = apg_points(apg,peak,w, y, z)
    apg_p = [a, b, c, d, e]
    apg_p_names = ['a', 'b', 'c', 'd', 'e']
    
    #apg amplitudes
    # apg amplitudes
    feats += [safe_idx(apg, i) for i in apg_p]
    feats_header += [f'apg_{n}' for n in apg_p_names]

    # ppg amplitudes
    feats += [safe_idx(cycle, i, offset=2) if i is not None else np.nan for i in apg_p]
    feats_header += [f'ppg_{n}' for n in apg_p_names]

    
    # ratio apg
    apg_a = safe_idx(apg, a)
    feats += [safe_div(safe_idx(apg, i), apg_a) for i in apg_p[1:]]
    feats_header += [f'ratio_apg_{n}' for n in apg_p_names[1:]]

    # ratio ppg
    ppg_a = safe_idx(cycle, a, offset=2) if a is not None else np.nan
    feats += [safe_div(safe_idx(cycle, i, offset=2), ppg_a) for i in apg_p[1:]]
    feats_header += [f'ratio_ppg_{n}' for n in apg_p_names[1:]]

    
    # Time apg points
    feats += [
        safe_div(a, fs),
        safe_div(safe_diff(b, a), fs),
        safe_div(safe_diff(c, b), fs),
        safe_div(safe_diff(d, c), fs),
        safe_div(safe_diff(e, d), fs),
    ]
    feats_header += [f'T_{n}' for n in apg_p_names]

    feats += [
        safe_div(a, Tc),
        safe_div(safe_diff(b, a), Tc),
        safe_div(safe_diff(c, b), Tc),
        safe_div(safe_diff(d, c), Tc),
        safe_div(safe_diff(e, d), Tc),
    ]
    feats_header += [f'T_{n}_norm' for n in apg_p_names]

    #Time apg points 2
    feats += [safe_div(safe_diff(peak,a), fs), safe_div(safe_diff(peak,b), fs), safe_div(safe_diff(c,peak), fs), safe_div(safe_diff(d,peak), fs), safe_div(safe_diff(e,peak), fs)]
    feats_header += ['T_peak_'+i for i in apg_p_names]
    
    feats += [safe_div(safe_diff(peak,a), Tc), safe_div(safe_diff(peak,b), Tc), safe_div(safe_diff(c,peak), Tc), safe_div(safe_diff(d,peak), Tc), safe_div(safe_diff(e,peak), Tc)]
    feats_header += ['T_peak_'+i+'_norm' for i in apg_p_names]
    # Aging Index
    AI = safe_div(
        safe_diff(safe_diff(safe_diff(safe_idx(apg, b), safe_idx(apg, c)), safe_idx(apg, d)), safe_idx(apg, e)),
        safe_idx(apg, a)
    )
    feats += [AI]
    feats_header += ['AI']

    # Others ratios
    if None in [b, d] or d == b:
        bd = np.nan
    else:
        bd = safe_div(
            safe_idx(vpg, d, offset=1) - safe_idx(vpg, b, offset=1),
            d - b
        )

    bcda = safe_div(
        safe_idx(apg, b) - safe_idx(apg, c) - safe_idx(apg, d),
        safe_idx(apg, a)
    )

    if peak is None or d is None or peak >= d:
        sdoo = np.nan
    else:
        num = np.sum(vpg[peak:d+1] ** 2)
        den = np.sum(vpg ** 2)
        sdoo = safe_div(num, den)

    feats += [bd, bcda, sdoo]
    feats_header += ['bd', 'bcda', 'sdoo']

    feats += [e]
    feats_header += ['e']
    return feats_header, feats

import numpy as np

def is_invalid(x):
    """None または np.nan を True"""
    return x is None or (isinstance(x, float) and np.isnan(x))

def safe_idx(arr, idx, offset=0):
    """
    idx が None / np.nan / 範囲外なら np.nan
    offset は idx が有効なときだけ足す
    """
    if is_invalid(idx):
        return np.nan

    j = int(idx) + offset  # idx が float の場合に備えて int
    if j < 0 or j >= len(arr):
        return np.nan

    val = arr[j]
    if is_invalid(val):
        return np.nan

    return val


def safe_div(num, den):
    """None / np.nan / 0 を含む割り算を np.nan に"""
    if is_invalid(num) or is_invalid(den):
        return np.nan
    if den == 0:
        return np.nan
    return num / den


def safe_diff(x, y):
    """x - y を安全に（None / nan 対応）"""
    if is_invalid(x) or is_invalid(y):
        return np.nan
    return x - y


def extract_temp_feat(cycle, peak, fs,ver2=False):
    """
    Extract temporal features related to interest points of VPG & APG of one cycle PPG.

    Parameters
    ----------
    cycle: array
        PPG cycle waveform. 
    peak: int
        Index of the systolic peak.
    fs: int
        Frequency sampling rate (Hz)

    Returns
    -------
    array
        Header with a list of the name of each feature. 
    array
        Extracted features related to interest points of VPG & APG of one cycle PPG.
    """
    
    # REMEMBER TO DIVIDE BY FS
    feat = []
    
    #Time of the cycle
    Tc = len(cycle)
    #Time from start to sys peak
    Ts = peak
    
    #Time from systolic to end
    Td = len(cycle) - peak
    
    vpg = np.diff(cycle)
    if ver2:
        vpg2 = np.convolve(vpg*1000, np.ones(6) / 6, mode='same')
        w,y,z = vpg_points2(vpg2,peak)
    else:
        w, y, z = vpg_points(vpg, peak)
    
    feats_header_apg, feats_apg = extract_apg_feat(cycle, vpg, peak, w, y, z, fs,ver2=ver2)
    
    if z is None:
        # print("Z is None: replacing with e",feats_apg[-1],feats_apg,len(feats_apg))
        z = feats_apg[-1]
    feats_apg = feats_apg[:-1]  # z は除く
    feats_header_apg = feats_header_apg[:-1]
    #Time from cycle start to first peak in VPG (steepest point)
    #steepest point == max value between start and peak
    Tsteepest = w
    Steepest = safe_idx(vpg, w)
    
    TNegSteepest = y
    
    #TdiaRise. Max positive slope after 'peak'
        
    #Greatest negative steepest (slope) from peak to end. (Slope, Time)
    NegSteepest = safe_idx(vpg, TNegSteepest)
    TdiaRise = z
    # Amplitude to DiaRise
    # print("TdiaRise",TdiaRise)
    DiaRise = safe_idx(cycle, TdiaRise)
    # SlopeDiaRise
    SteepDiaRise = safe_idx(vpg, TdiaRise)
        
    #Time from Systolic peak to Diastolic Rise
    TSystoDiaRise = safe_diff(TdiaRise, Ts)
    
    #Time from Diastolic Rise to End
    TdiaToEnd = safe_diff(Tc, TdiaRise)
    
    #Ratio between systolic peak and diastolic rise amplitude
    Ratio = cycle[peak]/DiaRise
    
    point_feat_name = ['Tc', 'Ts', 'Td', 'Tsteepest', 'Steepest', 'TNegSteepest', 'NegSteepest', 
            'TdiaRise', 'DiaRise', 'SteepDiaRise', 'TSystoDiaRise', 'TdiaToEnd', 'Ratio']
    point_feat = [safe_div(Tc, fs), safe_div(Ts, fs), safe_div(Td, fs), safe_div(Tsteepest, fs), Steepest, safe_div(TNegSteepest, fs), NegSteepest, 
            safe_div(TdiaRise, fs), DiaRise, SteepDiaRise, safe_div(TSystoDiaRise, fs), safe_div(TdiaToEnd, fs), Ratio]
    
    #norm by cycle
    point_feat_name = point_feat_name + ['Ts_norm', 'Td_norm', 'Tsteepest_norm', 'TNegSteepest_norm',
                       'TdiaRise_norm', 'TSystoDiaRise_norm', 'TdiaToEnd_norm']
    point_feat = point_feat + [safe_div(Ts, Tc), safe_div(Td, Tc), safe_div(Tsteepest, Tc), safe_div(TNegSteepest, Tc), 
            safe_div(TdiaRise, Tc), safe_div(TSystoDiaRise, Tc), safe_div(TdiaToEnd, Tc)]
    
    #width_at_per
    width_names = []
    width_feats = []

    for per in [0.25,0.50,0.75]:
        SW, DW = width_at_per(per, cycle, peak, fs)
        per_str = str(int(per*100))
        width_names += ['SW'+per_str, 'SW'+per_str+'_norm', 
                               'DW'+per_str, 'DW'+per_str+'_norm', 
                               'SWaddDW'+per_str, 'SWaddDW'+per_str+'_norm',
                               'DWdivSW'+per_str]
        width_feats +=[safe_div(SW, fs), safe_div(SW, Tc),
                       safe_div(DW, fs), safe_div(DW, Tc),
                        safe_div(SW+DW, fs), safe_div(SW+DW, Tc),
                        safe_div(DW, SW)]
        
    point_feat_name += width_names
    point_feat += width_feats
    
    min_val=np.min(cycle)
    #Area under the curve (AUC) from start of cycle to max upslope point
    S1 = np.trapz(cycle[:Tsteepest]-min_val)
    #AUC from max upslope point to systolic peak
    S2 = np.trapz(cycle[Tsteepest:peak]-min_val)
    #AUC from systolic peak to diastolic rise 
    AUCsys = S1+S2
    # print("TdiaRise in extract_temp_feat:",TdiaRise,feats_apg[-1])
    if is_invalid(TdiaRise):
        S3 = np.trapz(cycle[peak:]-min_val)
        AUCdia = S3
        S4 = np.nan
    else: 
        S3 = np.trapz(cycle[peak:TdiaRise]-min_val)
        #AUC from diastolic rise to end of cycle
        S4 = np.trapz(cycle[TdiaRise:]-min_val)
        #AUC of systole area S1+S2
        #AUC of diastole area S3+S4
        AUCdia = S3+S4
    area_feat_name = ['S1','S2','S3','S4','AUCsys','AUCdia']
    area_feat = [S1,S2,S3,S4,AUCsys,AUCdia]
    
    area_feat_name += ['S1_norm','S2_norm','S3_norm','S4_norm','AUCsys_norm','AUCdia_norm']
    area_feat += [S1/AUCsys,S2/AUCsys,safe_div(S3,AUCdia),safe_div(S4,AUCdia),AUCsys/(AUCsys+AUCdia),AUCdia/(AUCsys+AUCdia)]
    
    # SQI feats
    SQI_skew = skew(cycle,0.3)
    SQI_kurtosis = kurtosis(cycle)
    sqi_feat_name = ['SQI_skew','SQI_kurtosis']
    sqi_feat = [SQI_skew,SQI_kurtosis]
    
    feat_name = point_feat_name + area_feat_name +sqi_feat_name + feats_header_apg
    feat = point_feat + area_feat + sqi_feat + feats_apg
    

    # print(feat_name, feat)
    return np.array(feat_name),np.array(feat)


### ---------- Previous Feature Extraction Functions ----------

def signal_fft(data, fs, norm='ortho'):
    """Get the frequency range of signal using FFT
     We can get signal frequency plot with plt.plot(freq, abs_org_fft)

    Parameters
    ----------
    data : array
        1D array of the signal.
    fs : float
        Sampling rate of the signal.
    norm : {None, "ortho"}, default="ortho"
        Norm used to compute FFT

    Returns
    -------
    freq : array
        Discrete frequency range
    abs_org_fft : array
        Number of samples of each frequency
    """ 

    org_fft = np.fft.fft(data, norm=norm)
    abs_org_fft = np.abs(org_fft)
    freq = np.fft.fftfreq(data.shape[0], 1/fs)
    abs_org_fft = abs_org_fft[freq > 0]
    freq = freq[freq > 0]
    
    return freq, abs_org_fft

def get_fft_peaks(fft, freq, fft_peak_distance = 28, num_iter = 5):

    """ Extract the peaks of the signal's FFT given as parameter.

    Parameters
    ----------
    fft : array
        Number of samples of each frequency computed with `signal_fft`.
    freq : array
        Discrete frequency range computed with `signal_fft`.
    fft_peak_distance : int
        Minimum distance from peak to peak.
    num_iter : int
        Number of peaks to consider.

    See Also
    --------
    signal_fft

    Returns
    -------
    array
        Peaks of the signal's FFT.

    """ 
    
    peaks = scipy.signal.find_peaks(fft[:len(fft)//6], distance=fft_peak_distance)[0]# all of observed distance > 28
    # print("number of fft peaks found:",len(peaks),peaks)
    if len(peaks) > 1:
        peaks = peaks[peaks>fft_peak_distance]
    peaks = peaks[0:num_iter]
    return peaks

def fft_peaks_neighbor_avg(fft, fft_peaks, fft_neighbor_avg_interval = 6):
    """ Compute the average values nearby the fft_peaks.
    This feature is used mainly for PPG, but it can be used for ECG.

    Parameters
    ----------
    fft : array
        Number of samples of each frequency computed with `signal_fft`.
    fft_peaks : array
        Peaks of FFT computed with `get_fft_peaks`.
    fft_neighbor_avg_interval : int
        Minimum distance from peak to peak.
    fft_neighbor_avg_interval : int
        Range of neighbors to consider.

    See Also
    --------
    signal_fft, get_fft_peaks 

    Returns
    -------
    array
        Average values nearby the fft_peaks.

    """ 
    
    fft_peaks_neighbor_avgs = []
    for peak in fft_peaks:
        start_idx = peak - fft_neighbor_avg_interval
        end_idx = peak + fft_neighbor_avg_interval
        fft_peaks_neighbor_avgs.append(fft[start_idx:end_idx].mean())
    return np.array(fft_peaks_neighbor_avgs)

def extract_cycles_all_ppgs(waveforms, ppg_peaks, hr_offset, match_position, remove_first = True):
    """ Extract cycles from the waveforms of PPG and its derivatives.

    Parameters
    ----------
    waveforms : dict {"ppg": ppg, "vpg": vpg, "apg": apg, "ppg3": ppg3, "ppg4": ppg4}
        All the waveforms of PPG and its derivatives.
    ppg_peaks: array
        Peaks of the PPG signal.
    hr_offset: float
        Distance from peak to peak.
    match_position: {"sys_peak", "dia_notches"}
        Position to match the cycles.
        Systolic peaks ("sys_peak") or Diastolic notches ("dia_notches")

    Returns
    -------
    dict 
        {"ppg_cycles": array, "vpg_cycles": array, "apg_cycles": array, "ppg3_cycles": array, "ppg4_cycles": array}
        Cycles from the waveforms of PPG and its derivatives.
    """ 

    offset = np.round(hr_offset).astype("int")  # Compute the window cut

    # Extract cycle based on ppg_peaks

    waveforms_cycles = {
        "ppg_cycles": [],
        "vpg_cycles": [],
        "apg_cycles": [],
        "ppg3_cycles": [],
        "ppg4_cycles": []
    }

    # remove head and tail peaks
    if remove_first:
        ppg_peaks = ppg_peaks[1:-1] ## Original
    else: ## Addition for small waveforms
        if ppg_peaks[0] == 0:
            ppg_peaks = ppg_peaks[1:]
        if ppg_peaks[-1] == len(waveforms['ppg'])-1:
            ppg_peaks = ppg_peaks[:-1]
            
    
    lower_offset = offset * 0.25
    upper_offset = offset * 0.75

    for p in ppg_peaks:

        start = np.round(p - lower_offset).astype("int")
        end = np.round(p + upper_offset).astype("int")

        # Align two diastolic notches of cycles
        if match_position == "dia_notches":
            tolerance = 0.1

            start = np.round(p - offset * (0.25 + tolerance)).astype("int")
            end = np.round(p + offset * (0.75 + tolerance)).astype("int")
            
            # check range
            if (start < 0) or (p <= start) or \
               (int(p + offset * (0.75-tolerance)) < 0) or (end <= int(p + offset * (0.75-tolerance))) or \
               (len(waveforms["ppg"][start:p])) == 0 or len(waveforms["ppg"][int(p + offset * (0.75-tolerance)):end]) <= 0:
                continue
            
            # stand and end from valleys
            start = start + np.argmin(waveforms["ppg"][start:p])
            end = int(p + offset * (0.75-tolerance)) + np.argmin(waveforms["ppg"][int(p + offset * (0.75-tolerance)):end])

        if (start < 0) or (end > waveforms["ppg"].shape[0]):
            continue

        for waveform_name in waveforms:
            waveforms_cycles[waveform_name + "_cycles"].append(waveforms[waveform_name][start:end])

    for waveform_name in waveforms:
        waveforms_cycles[waveform_name + "_cycles"] = np.array(waveforms_cycles[waveform_name + "_cycles"], dtype=object)
        
    return waveforms_cycles

def mean_norm_cycles(cycles, resample_length = 80):
    """ Calculate mean of cycles which is normalized with min-max normalization and resampled.

    Parameters
    ----------
    cycles : array
        Cycles of the signal to compute the mean (2D-array)
    resample_length: int
        Resample length.

    Returns
    -------
    avg_normalized_cycles : array
        Normalized mean of the cycles.
    normalized_cycles : array
        Normalized and Resampled cycles.
    """ 
    
    normalized_cycles = []
    for cycle in cycles:
        normalized_cycle = scipy.signal.resample(cycle, resample_length)
        normalized_cycle = waveform_norm(normalized_cycle)
        normalized_cycles.append(normalized_cycle)

    if len(normalized_cycles) > 0:
        normalized_cycles = np.array(normalized_cycles)
        
    #avg_normalized_cycles = normalized_cycles.mean(axis=0)
    avg_normalized_cycles = np.median(normalized_cycles, axis=0)
    return avg_normalized_cycles, normalized_cycles

def max_neighbor_mean(mean_cycles, neighbor_mean_size = 5):
    """ Compute the mean of values near the maximum value.

    Parameters
    ----------
    mean_cycles : array
        Mean of all cycles of the signal.
    neighbor_mean_size: int
        Range of near values to consider in the average.

    Returns
    -------
    float 
        Mean of values near the maximum value.
    
    References
    ----------
    .. [1] https://www.researchgate.net/profile/Aman_Gaurav/publication/328994912_InstaBP_Cuff-less_Blood_Pressure_Monitoring_on_Smartphone_using_Single_PPG_Sensor/links/5bf68e3da6fdcc3a8de93629/InstaBP-Cuff-less-Blood-Pressure-Monitoring-on-Smartphone-using-Single-PPG-Sensor.pdf

    """
    
    ppg_start_idx = max(np.argmax(mean_cycles) - neighbor_mean_size, 0)
    ppg_end_idx = min(np.argmax(mean_cycles) + neighbor_mean_size, len(mean_cycles))

    if ppg_start_idx == ppg_end_idx: 
        ppg_end_idx += 1

    return mean_cycles[ppg_start_idx:ppg_end_idx].mean()

def histogram_up_down(mean_cycles, num_up_bins, num_down_bins, ppg_max_idx):
    """ Compute histogram features of the cycle given as parameter. 
    Two histograms are computed.
        - Up: From the start of the cycle to the maximum value.
        - Down: From the maximum value to the end of the cycle.


    Parameters
    ----------
    mean_cycles : array
        Cycle or Mean of all cycles of the signal.
    num_up_bins: int
        Number of bins for the Up histogram.
    num_down_bins: int
        Number of bins for the down histogram.
    ppg_max_idx: int
        Index markind the maximum value of the cycle.

    Returns
    -------
    H_up : array 
        Values of the Up histogram. 
    H_down: array 
        Values of the Down histogram.
    bins_up : array 
        Bin edges of the Up histogram.
    bins_down : array 
        Bin edges of the Down histogram.

    References
    ----------
    .. [1] https://www.researchgate.net/profile/Aman_Gaurav/publication/328994912_InstaBP_Cuff-less_Blood_Pressure_Monitoring_on_Smartphone_using_Single_PPG_Sensor/links/5bf68e3da6fdcc3a8de93629/InstaBP-Cuff-less-Blood-Pressure-Monitoring-on-Smartphone-using-Single-PPG-Sensor.pdf

    """
    
    H_up, bins_up = np.histogram(mean_cycles[:ppg_max_idx], bins=num_up_bins, range=(0,1), density=True)
    H_down, bins_down = np.histogram(mean_cycles[ppg_max_idx:], bins=num_down_bins, range=(0,1), density=True)

    return H_up, H_down, bins_up, bins_down

def USDC(cycles, USDC_resample_length):
    """ Compute mean feature with UpSlope Deviation curve. 
    Deviation of each point from the mean upslope on the rising edge 
    and depict the relative speed of systolic activity.

    This feature is designed for PPG cycles.
    
    Parameters
    ----------
    cycles : array
        Cycles of the signal.
    USDC_resample_length: int
        Resample of the systolic segment.

    Returns
    -------
    array 
        Average of the USDC features extracted for each cycle.

    References
    ----------
    .. [1] https://www.researchgate.net/profile/Aman_Gaurav/publication/328994912_InstaBP_Cuff-less_Blood_Pressure_Monitoring_on_Smartphone_using_Single_PPG_Sensor/links/5bf68e3da6fdcc3a8de93629/InstaBP-Cuff-less-Blood-Pressure-Monitoring-on-Smartphone-using-Single-PPG-Sensor.pdf
    """
    
    usdc_features = []
    for cycle in cycles:

        # calculate usdc of one cycle
        max_idx = np.argmax(cycle)
        cycle = scipy.signal.resample(cycle[:max_idx+1], USDC_resample_length)
        max_idx = len(cycle) - 1
        usdc = (cycle * (cycle[max_idx] - cycle[0]) * np.arange(len(cycle)) - cycle * max_idx + cycle[0] * max_idx) / (np.sqrt((cycle[max_idx] - cycle[0]) ** 2 + max_idx ** 2))

        # calculate usdc feature, similar to convolute on usdc
        interval = 3
        usdc_feature = []
        for idx in range(0, max_idx, interval):
            if idx+interval < len(usdc):
                usdc_feature.append(usdc[idx:idx+interval].mean())

        usdc_features.append(usdc_feature)

    usdc_features = np.array(usdc_features)
    mean_usdc_features = usdc_features.mean(axis=0)

    return mean_usdc_features

def DSDC(cycles, DSDC_resample_length):
    """ Compute mean feature with DownSlope Deviation curve. 
    Deviation of each point from the mean downslope on the falling edge 
    and depict the relative speed of diastolic activity.

    This feature is designed for PPG cycles.
    
    Parameters
    ----------
    cycles : array
        Cycles of the signal.
    DSDC_resample_length: int
        Resample of the diastolic segment.

    Returns
    -------
    array 
        Average of the DSDC features extracted for each cycle.

    References
    ----------
    .. [1] https://www.researchgate.net/profile/Aman_Gaurav/publication/328994912_InstaBP_Cuff-less_Blood_Pressure_Monitoring_on_Smartphone_using_Single_PPG_Sensor/links/5bf68e3da6fdcc3a8de93629/InstaBP-Cuff-less-Blood-Pressure-Monitoring-on-Smartphone-using-Single-PPG-Sensor.pdf
    """
    
    dsdc_features = []
    for cycle in cycles:

        # calculate dsdc of one cycle
        max_idx = np.argmax(cycle)
        cycle = scipy.signal.resample(cycle[max_idx:], DSDC_resample_length)
        l = len(cycle) - 1
        max_idx = 0
        dsdc = (cycle * (cycle[l] - cycle[max_idx]) * np.arange(len(cycle)) - (l - max_idx) * cycle + cycle[max_idx] * l - cycle[l] * max_idx) / np.sqrt((cycle[l] - cycle[max_idx]) ** 2 + (l - max_idx) ** 2)

        # calculate dsdc feature, similar to convolute on dsdc
        interval = 3
        dsdc_feature = []
        for idx in range(max_idx, len(dsdc), interval):
            if idx+interval < len(dsdc):
                dsdc_feature.append(dsdc[idx:idx+interval].mean())

        dsdc_features.append(dsdc_feature)

    dsdc_features = np.array(dsdc_features)
    mean_udsc_features = dsdc_features.mean(axis=0)

    return mean_udsc_features

def generate_features_csv_string(features):
    """ Transform a dictionary of extracted features to a string in CSV format.
    
    Parameters
    ----------
    features : dict
        Dictionary of the extracted features with their names and values.

    Returns
    -------
    header : list 
        Header of with the names of the different features.
        [{feature_name_0}, {feature_name_1}, {feature_name_2}...] - Features with the same name will add _1 _2 _3 as suffix, _0 is a fixed suffix.
    features_csv : str 
        Values of the features in csv string format.

    """
    
    header = []
    features_csv = []

    for feature_name in features:
        feature = features[feature_name]
        # print(feature_name)
        try:
            len_feature = len(feature)
        except:
            len_feature = 1

        for i in range(len_feature):
            header.append(feature_name + "_" + str(i))
        features_csv.append(feature)

    # flatten list
    features_csv = list(np.hstack(features_csv))
    
    features_csv = str(features_csv).strip("[]")

    return header, features_csv



### ---------- Implementation of PPG ----------

class PPG:
    """ Implementation PPG class for features extraction of PPG signals.

    Parameters
    ----------
    data : array
        PPG signal 1D-array.
    fs : int
        Sampling rate.
    cycle_size : int
        Resampling length when the cycles are extracted.
        
    Attributes
    ----------
    data : array
        PPG signal array.
    idata : array
        Inverse of the signal used to compute the valleys.
    fs : array
        Sampling rate.
    cycle_size : int
        Resampling length when the cycles are extracted.

    """
    def __init__(self, data, fs, cycle_size=128):
        self.data = data
        self.idata = data.max() - data
        self.fs = fs
        self.cycle_size = cycle_size

    def peaks(self, **kwargs): # systolic peaks
        """ Extract the peaks of the PPG signal 

        Returns
        -------
        array 
            Indeces marking the extracted peaks.
        """
        # x: ppg signal
        return pyampd.ampd.find_peaks(self.data, scale=int(self.fs))

    def vpg(self, **kwargs):
        """ Compute the 1st Derivative of the PPG.

        Returns
        -------
        array 
            1st Derivative of the PPG signal (vpg).
        """
        vpg = self.data[1:] - self.data[:-1]
        padding = np.zeros(shape=(1))
        vpg = np.concatenate([padding, vpg], axis=-1)
        return vpg

    def apg(self, **kwargs):
        """ Compute the 2nd Derivative of the PPG.

        Returns
        -------
        array 
            2nd Derivative of the PPG signal (apg).
        """
        apg = self.data[1:] - self.data[:-1]  # 1st Derivative
        apg = apg[1:] - apg[:-1]  # 2nd Derivative

        padding = np.zeros(shape=(2))
        apg = np.concatenate([padding, apg], axis=-1)
        return apg
    
    def ppg3(self, **kwargs):
        """ Compute the 3rd Derivative of the PPG.

        Returns
        -------
        array 
            3rd Derivative of the PPG signal.
        """
        ppg3 = self.data[1:] - self.data[:-1]  # 1st Derivative
        ppg3 = ppg3[1:] - ppg3[:-1]  # 2nd Derivative
        ppg3 = ppg3[1:] - ppg3[:-1]  # 3nd Derivative

        padding = np.zeros(shape=(3))
        ppg3 = np.concatenate([padding, ppg3], axis=-1)
        return ppg3
    
    def ppg4(self, **kwargs):
        """ Compute the 4th Derivative of the PPG.

        Returns
        -------
        array 
            4th Derivative of the PPG signal.
        """
        ppg4 = self.data[1:] - self.data[:-1]  # 1st Derivative
        ppg4 = ppg4[1:] - ppg4[:-1]  # 2nd Derivative
        ppg4 = ppg4[1:] - ppg4[:-1]  # 3nd Derivative
        ppg4 = ppg4[1:] - ppg4[:-1]  # 4nd Derivative

        padding = np.zeros(shape=(4))
        ppg4 = np.concatenate([padding, ppg4], axis=-1)
        return ppg4

    def hr(self, **kwargs):
        """ Compute the heart rate from the peak distances in BPM, returns 0 if peaks are not present.
        Returns
        -------
        float 
            Heart rate in BPM of the signal.
        """ 
        try: return self.fs / np.median(np.diff(self.peaks())) * 60
        except: return 0

    def diastolic_notches(self, **kwargs):
        """ Extract the diastolic notches of the PPG signal.
        Similar to `valleys` function.

        Returns
        -------
        array 
            Indeces marking the diastolic notches.
        """
        notches = pyampd.ampd.find_peaks(-self.data, scale=int(self.fs))
        #notches = scipy.signal.find_peaks(-self.data, distance=35, height=-self.data.mean())[0]
        
        return notches    
       
    def features_extractor(self, filtered=False, 
        fft_peak_distance = 28, fft_neighbor_avg_interval = 6, 
        resample_length = 80,
        neighbor_mean_size = 5,
        num_up_bins = 5,
        num_down_bins = 10, 
        remove_first = True,
        one_cycle_sig = False):
        """ Extract all features from PPG

        Returns
        -------
        features : dict 
            Extracted features with their names and values as {"{feature_name}": feature_value, ...}
        header: list
            Names of the different features as:
            [{feature_name_0}, {feature_name_1}, {feature_name_2}...] - Features with the same name will add _1 _2 _3 as suffix, _0 is a fixed suffix
        features_csv_str : str
            Values of features in csv string format.

        Examples
        --------
        >>> ppg = PPG(ppg, 125)
        >>> features, header, features_csv_str = ppg.features_extractor()
        """
        
        if one_cycle_sig:
            remove_first = False
        
        features = {}
        
        # default parameters
        USDC_resample_length = resample_length // 4
        DSDC_resample_length = resample_length // 4 * 3

        # normalize ppg, vpg, apg, ppg3, ppg4
        ppg = self.data
        ppg = waveform_norm(ppg)
        if filtered:
            vpg = mean_filter_normalize( self.vpg(), int(self.fs), 0.75, 10, 1)
            apg = mean_filter_normalize( self.apg(), int(self.fs), 0.75, 10, 1)
            ppg3 = mean_filter_normalize( self.ppg3(), int(self.fs), 0.75, 10, 1)
            ppg4 = mean_filter_normalize( self.ppg4(), int(self.fs), 0.75, 10, 1)
        else:
            vpg = waveform_norm(self.vpg())
            apg = waveform_norm(self.apg())
            ppg3 = waveform_norm(self.ppg3())
            ppg4 = waveform_norm(self.ppg4())
            
        # get peaks and valleys
        sys_peaks = pyampd.ampd.find_peaks(self.data, scale=int(self.fs))
        dia_notches = self.diastolic_notches()
        
        if one_cycle_sig:
            sys_peaks = np.array([self.data.argmax()])
            dia_notches = np.array([0,len(self.data)-1])
        
        # all signals from a ppg signal
        waveforms = {
            "ppg": ppg,
            "vpg": vpg,
            "apg": apg,
            "ppg3": ppg3,
            "ppg4": ppg4
        }
        
        # Frequency domain features
        for waveform_name in waveforms:
            # print(waveform_name)
            freq, fft = signal_fft(waveforms[waveform_name], self.fs)
            fft = fft / np.linalg.norm(fft)
            fft_peaks = get_fft_peaks(fft, freq, fft_peak_distance, num_iter=5)
            # print("number of fft peaks:",len(fft_peaks))
            fft_peaks_neighbor_avgs = fft_peaks_neighbor_avg(fft, fft_peaks, fft_neighbor_avg_interval)
            features[waveform_name + "_fft_peaks"] = fft_peaks
            features[waveform_name + "_fft_peaks_heights"] = fft[fft_peaks]
            features[waveform_name + "_fft_peaks_neighbor_avgs"] = fft_peaks_neighbor_avgs
        # print("after fft", features.keys())
        # Time domain features
        p2p = np.median(np.diff(sys_peaks))
        if one_cycle_sig:
            p2p = len(self.data)
        
        waveforms_cycles_match_peak = extract_cycles_all_ppgs(waveforms, sys_peaks, p2p, "sys_peak", remove_first)
        waveforms_cycles_match_valleys = extract_cycles_all_ppgs(waveforms, sys_peaks, p2p, "dia_notches", remove_first)
        
        if one_cycle_sig:
            waveforms_cycles_match_valleys = {
                "ppg_cycles": [],
                "vpg_cycles": [],
                "apg_cycles": [],
                "ppg3_cycles": [],
                "ppg4_cycles": []
            }
            for waveform_name in waveforms:
                waveforms_cycles_match_valleys[waveform_name + "_cycles"] = np.array([waveforms[waveform_name]], dtype=object)
            
            waveforms_cycles_match_peak = copy.deepcopy(waveforms_cycles_match_valleys)
            
        
        if len(waveforms_cycles_match_peak["ppg_cycles"]) == 0:
            raise RuntimeError("Feature extractor warning: There are no cycles")
        if len(waveforms_cycles_match_valleys["ppg_cycles"]) == 0:
            raise RuntimeError("Feature extractor warning: There are no cycles 2")
        
        # HR and peak to peak distance
        #features["hr"] = self.hr()
        features["hr"] = self.fs / p2p * 60 
        features["p2p"] = p2p
        
        # average avg of signals nearby max. 
        # min positions
        for waveform_name in waveforms:
            cycles = waveforms_cycles_match_peak[waveform_name + "_cycles"]
            mean_cycles, norm_cycles = mean_norm_cycles(cycles, resample_length)
            if "ppg" == waveform_name:
                features["ppg_mean_cycles_match_peak"] = mean_cycles
            neighbor_mean = max_neighbor_mean(mean_cycles, neighbor_mean_size)
            features[waveform_name + "_max_neighbor_mean"] = neighbor_mean
            features[waveform_name + "_min"] = np.argmin(mean_cycles)
            
        # calculate ppg_max_idx
        ppg_cycles_match_valleys = waveforms_cycles_match_valleys["ppg_cycles"]
        ppg_mean_cycles_match_valleys, ppg_norm_cycles_match_valleys = mean_norm_cycles(ppg_cycles_match_valleys, resample_length)
        ppg_max_idx = np.argmax(ppg_mean_cycles_match_valleys)
        
        # average avg of signals nearby max. 
        for waveform_name in waveforms:
            cycles = waveforms_cycles_match_valleys[waveform_name + "_cycles"]
            mean_cycles, norm_cycles = mean_norm_cycles(cycles, resample_length)
            if "ppg" == waveform_name:
                features["ppg_mean_cycles_match_valleys"] = mean_cycles
            H_up, H_down, bins_up, bins_down = histogram_up_down(mean_cycles, num_up_bins, num_down_bins, ppg_max_idx)
            features[waveform_name + "_histogram_up"] = H_up
            features[waveform_name + "_histogram_down"] = H_down
            features[waveform_name + "_max"] = np.argmax(mean_cycles)
        
        # using cycles_match_peak to fix len of features
        usdc = USDC(ppg_norm_cycles_match_valleys, USDC_resample_length)
        dsdc = DSDC(ppg_norm_cycles_match_valleys, DSDC_resample_length)
        features["usdc"] = usdc
        features["dsdc"] = dsdc
        
        # generate header and csv
        header, features_csv_str = generate_features_csv_string(features)
        # print("header:",header)
        # print("features:",features.keys())
        # print(features["ppg_fft_peaks"])
        return features, header, features_csv_str
        


"""
The Signal Quality Index module provides multiple metrics to assess the quality of ECG and PPG signals. These scores can be used to filter the bad-quality signals. 

"""

import numpy as np

def _is_flat(x, threshold=0.3):
    """ Identify if a signal has flat regions greater than a given threshold

    Parameters
    ----------
    x : array
        Raw signal
    threshold : float
        Threshold of percentage of flat region [0-1].
    
    Returns
    -------
    bool
        True if the signal is flat for longer than the threshold

    Notes
    -----
    threshold=0.3 is generally recomended.
    """

    delta=1e-5
    dx = x[1:] - x[:-1]
    flat_parts = (np.abs(dx) < delta).astype("int").sum()
    flat_parts = flat_parts/x.shape[0]
    
    # Reject the segment if 30% of the signals are flat
    return flat_parts > threshold


def skew(x, is_flat_threshold): 
    """ Skewness of the signal given as parameter.
    Skewness measures of the symmetry of a probability distribution. It is associated with 
    Implemention of different metrics to assess the quality of a signal (PPG or ECG).

    Parameters
    ----------
    x : array
        Target signal, of which we want to measure the SQI.  
    is_flat_threshold : float
        Function parameters. kwargs['is_flat_threshold'] is the threshold of percentage of flat region.

    Returns
    -------
    float
        SQI skewness of the target signal.

    Notes
    -----
    threshold=0.3 can get normal boxplot. See more details in comments for function is_flat.
    Higher values of Skewness usually represent better quality.
    It is recommended not to exceed a signal window 3-5 seconds.

    References
    ----------
    .. [1] Elgendi, M. (2016). Optimal signal quality index for photoplethysmogram signals. Bioengineering, 3(4), 21. (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5597264/)
    """
    return np.sum(((x - x.mean())/(x.std()+1e-6))**3)/x.shape[0] * ~_is_flat(x, threshold=is_flat_threshold)


def kurtosis(x): 
    """ Kurtosis of the signal given as parameter. 
    Kurtosis describes the distribution of observed data around the mean.

    Parameters
    ----------
    x : array
        Raw signal, of which we want to measure the SQI.
    
    Returns
    -------
    float
        SQI kurtosis of the target signal.
    
    References
    ----------
    .. [1] Elgendi, M. (2016). Optimal signal quality index for photoplethysmogram signals. Bioengineering, 3(4), 21. (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5597264/)
    """

    sqi_score = 1/(np.sum(((x - x.mean())/(x.std()+1e-6))**4)/(x.shape[0]+1e-6) + 1e-6) # reciprocal kurtosis

    if np.isnan(sqi_score) or np.isinf(sqi_score):
        return 0.0

    return sqi_score

import pandas as pd
import numpy as np
import math


from pyampd.ampd import find_peaks

import scipy.signal
from scipy.signal import correlate
from scipy.interpolate import CubicSpline

def normalize_data(x):
    """Min-max normalization of signal waveform.

    Parameters
    ----------
    x : array
        Signal waveform.

    Returns
    -------
    array
        Normalized signal waveform.
    """
    return (x - x.min()) / (x.max() - x.min() + 1e-10)  # 1e-10 avoid division by zero

def waveform_norm(x):
    """Min-max normalization of signal waveform.

    Parameters
    ----------
    x : array
        Signal waveform.

    Returns
    -------
    array
        Normalized signal waveform.
    """
    return (x - x.min())/(x.max() - x.min() + 1e-6)


def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    """ Butterworth band-pass filter
    Parameters
    ----------
    data : array
        Signal to be filtered.
    lowcut : float
        Frequency lowcut for the filter. 
    highcut : float}
        Frequency highcut for the filter.
    fs : float
        Sampling rate.
    order: int
        Filter's order.

    Returns
    -------
    array
        Signal filtered with butterworth algorithm.
    """  
    nyq = fs * 0.5  # https://en.wikipedia.org/wiki/Nyquist_frequency
    lowcut = lowcut / nyq  # Normalize
    highcut = highcut / nyq
    # Numerator (b) and denominator (a) polynomials of the IIR filter
    b, a = scipy.signal.butter(order, [lowcut, highcut], btype='band', analog=False)
    return scipy.signal.filtfilt(b, a, data)


def remove_mean(data):
    """Remove the mean from the signal.

    Parameters
    ----------
    x : array
        Signal waveform.

    Returns
    -------
    array
        Processed signal waveform.
    """
    return data-np.mean(data)


def mean_filter_normalize(data, fs, lowcut, highcut, order):
    """
    Wrapper for removing mean, bandpass filter and normalizing the signals between 0-1

    Parameters
    ----------
    x : array
        Signal waveform.
    fs : int
        Sampling rate.
    lowcut : float
        Frequency lowcut for the filter. 
    highcut : float}
        Frequency highcut for the filter.
    order: int
        Filter's order.

    Returns
    -------
    array
        Processed signal waveform.
    """
    data = data-np.mean(data)
    data = butter_bandpass_filter(data, lowcut, highcut, fs, order)
    data = normalize_data(data)
    
    return data


def align_pair(abp, raw_ppg, windowing_time, fs):
    """
    Align ABP and PPG signal passed as parameters using the maximum cross-correlation.
    Only PPG is shifted to align with ABP. The shift is limited to a second as maximum.

    Parameters
    ----------
    abp : array
        ABP signal waveform
    raw_ppg : array
        PPG signal waveform
    windowing_time: int
        Length of the signals in seconds
    fs: int
        Frequency sampling rate (Hz)
    
    Returns
    -------
    array
        Aligned ABP signal waveform
    array
        Aligned PPG signal waveform
    Int
        Number of samples shifted

    """

    window_size = fs * windowing_time # original segment length
    extract_size = fs * (windowing_time-1)

    cross_correlation = correlate(abp, raw_ppg)
    shift = np.argmax(cross_correlation[extract_size:window_size]) #shift must happened within 1s
    shift += extract_size
    start = np.abs(shift-window_size)

    a_abp = abp[:extract_size]
    a_rppg = raw_ppg[start:start+extract_size]

    return a_abp, a_rppg, shift-window_size


def rm_baseline_wander(ppg, vlys, add_pts = True):
    """
    Remove baseline wander (BW) from a signal subtracting an BW estimated with Cubic Spline.

    Parameters
    ----------
    ppg : array
        signal waveform to process.
    vlys: array
        Indices of the valleys of the signal.
    add_pts: bool
        Enable to add points to cover all the signal.
    
    Returns
    -------
    array
        Processed signal waveform without BW.
    array
        Estimated baseline wander.
    array
        Values used to estimate BW.
    array
        Indices of the values used to estimate BW.
    """    

    rollingmin_idx = vlys
    rollingmin = ppg[vlys]
    
    mean = np.mean(rollingmin)
    
    if add_pts == True:
        dist = np.median(np.diff(rollingmin_idx))
        med = np.median(rollingmin)
        
        add_pts_head = math.ceil(rollingmin_idx[0] / dist)
        head_d = [rollingmin_idx[0]-i*dist for i in reversed(range(1,add_pts_head+1))] 
        head_m = [med]*add_pts_head
        
        
        add_pts_tail = math.ceil((len(ppg)-rollingmin_idx[-1]) / dist)
        tail_d = [rollingmin_idx[-1]+ i*dist for i in range(1,add_pts_tail+1)] 
        tail_m = [med]*add_pts_tail 
        
        rollingmin_idx = np.concatenate((head_d, rollingmin_idx, tail_d))
        rollingmin = np.concatenate((head_m, rollingmin, tail_m))
    # polyfit

    cs = CubicSpline(rollingmin_idx, rollingmin)

    # polyval

    baseline = cs(np.arange(len(ppg)))

    # subtract the baseline
    
    rem_line = ppg - (baseline-mean)

    return rem_line, baseline, rollingmin, rollingmin_idx


def identify_out_pk_vly(sig, pk, vly, th=3):
    """
    Identify outliers in the peaks and valleys of the signal passed as parameters.
    Peak or valley is an outlier if it exceeds 'th' times the standard deviation w.r.t. the mean of the peaks/valleys.

    Parameters
    ----------
    sig : array
        signal waveform.
    pk: array
        Indices of the peaks of the signal.
    vly: array
        Indices of the valleys of the signal.
    th: float
        Threshold to identify outliers
    
    Returns
    -------
    list
        Indices of the identified outliers
    """ 

    out_pk, out_vly = -1, -1
    outs = []
    
    vly_val = sig[vly]
    pk_val = sig[pk]
    
    vly_val_idx = vly_val.argmin()
    vly_val_min = vly_val[vly_val_idx]
    vly_val_argmin = vly[vly_val_idx]
    vly_val_mean = vly_val.mean()
    vly_val_std = vly_val.std()
    
    if vly_val_min < vly_val_mean - vly_val_std * th:
        outs.append(vly_val_argmin)
    
    pk_val_idx = pk_val.argmax()
    pk_val_max = pk_val[pk_val_idx]
    pk_val_argmax = pk[pk_val_idx]
    pk_val_mean = pk_val.mean()
    pk_val_std = pk_val.std()
    
    if pk_val_max > pk_val_mean + pk_val_std * th:
        outs.append(pk_val_argmax)
    
    return outs


def my_find_peaks(sig, fs, remove_start_end = True):
    """ 
    Wrapper of find peaks function. If find_peaks fails, an empty array is returned.

    Parameters
    ----------
    sig : array
        signal waveform.
    fs : int
        Sampling rate.
    remove_start_end: bool
        Indices of the valleys of the signal.

    Returns
    -------
    array
        Indices of the peaks found.

    """

    try:
        pks = pyampd.ampd.find_peaks(sig, scale=int(fs))
        if remove_start_end:
            if pks[0] == 0: pks = pks[1:]

            if pks[-1] == len(sig)-1: pks = pks[:-1]
        return pks
    except:
        return np.array([])






