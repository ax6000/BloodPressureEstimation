import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def segment_ppg(ppg_signal, sampling_rate=125):
    """
    PPG信号を特徴的な部分（ピーク、ノッチ、谷）で分割する
    
    Parameters:
    -----------
    ppg_signal : array-like
        PPG信号データ
    sampling_rate : int
        サンプリングレート（デフォルト: 125Hz）
    
    Returns:
    --------
    dict
        各特徴点のインデックスを含む辞書
    """
    # ピークの検出
    peaks, _ = find_peaks(ppg_signal, distance=sampling_rate//2, prominence=0.1)
    
    # 谷の検出（信号を反転させてピーク検出）
    valleys, _ = find_peaks(-ppg_signal, distance=sampling_rate//2, prominence=0.1)
    
    # ノッチの検出（ピークと谷の間の小さなくぼみ）
    # ピークと谷の間の領域でローカルな極小値を探す
    notches = []
    for i in range(len(peaks)-1):
        if i < len(valleys):
            # ピークと次の谷の間の区間
            start_idx = peaks[i]
            end_idx = valleys[i]
            if start_idx < end_idx:
                segment = ppg_signal[start_idx:end_idx]
                # その区間での小さなくぼみ（ノッチ）を検出
                local_minima, _ = find_peaks(-segment, prominence=0.05)
                if len(local_minima) > 0:
                    notches.append(start_idx + local_minima[0])
    
    return {
        'peaks': peaks,
        'valleys': valleys,
        'notches': np.array(notches)
    }

def visualize_segmentation(ppg_signal, features, sampling_rate=125):
    """
    PPG信号の分割結果を可視化する
    
    Parameters:
    -----------
    ppg_signal : array-like
        PPG信号データ
    features : dict
        segment_ppg関数から返される特徴点の辞書
    sampling_rate : int
        サンプリングレート（デフォルト: 125Hz）
    """
    time = np.arange(len(ppg_signal)) / sampling_rate
    
    plt.figure(figsize=(15, 5))
    plt.plot(time, ppg_signal, label='PPG Signal')
    plt.plot(time[features['peaks']], ppg_signal[features['peaks']], 'ro', label='Peaks')
    plt.plot(time[features['valleys']], ppg_signal[features['valleys']], 'go', label='Valleys')
    plt.plot(time[features['notches']], ppg_signal[features['notches']], 'mo', label='Notches')
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.title('PPG Signal Segmentation')
    plt.legend()
    plt.grid(True)
    plt.show()

# 使用例
if __name__ == "__main__":
    # サンプルのPPG信号を生成（実際のデータに置き換えてください）
    t = np.linspace(0, 10, 1250)
    sample_ppg = np.sin(2 * np.pi * 1.0 * t) + 0.5 * np.sin(2 * np.pi * 2.0 * t)
    
    # 信号の分割
    features = segment_ppg(sample_ppg)
    
    # 結果の可視化
    visualize_segmentation(sample_ppg, features) 