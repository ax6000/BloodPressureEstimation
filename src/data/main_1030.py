import numpy as np
import glob
import os
import logging
from tqdm import tqdm

from preprocess import flat,butterworth,hampel,minmax_norm

def pipeline(sig):
    sig = filter_ppg(sig)
    sig = flat(sig,plot=False)
    return sig

def pipeline2(sig):
    sig[:,1] = minmax_norm(sig[:,1])
    sig = flat(sig,plot=False)
    
def filter_ppg(sig):
    if sig.shape[1] != 2:
        logging.info(f"invalid because of shape: {sig.shape}")
        return None  
    notnan= ~np.isnan(sig[:,1]) 
    s = np.sum(notnan)
    if s < 125*120:
        logging.info(f"invalid because of short valid signal: {s}")
        return None
    
    # print(np.sum(notnan))
    sig[notnan,1] = butterworth(sig[notnan,1])
    sig[notnan,1] = hampel(sig[notnan,1])
    sig[:,1] = minmax_norm(sig[:,1])
    return sig

def main():
    DATA_DIR = r"D:\minowa\BloodPressureEstimation\data\raw\ppgabp\p00"
    OUTPUT_DIR = r"D:\minowa\BloodPressureEstimation\data\processed\npy\1030\p00"
    logging.basicConfig(filename=os.path.join(OUTPUT_DIR,'output.log'), level=logging.INFO)
    files = glob.glob("**\*.npy",recursive=True,root_dir=DATA_DIR)
    cnt = 0
    for i in tqdm(range(len(files))):
        sig = np.load(os.path.join(DATA_DIR,files[i]))
        sig = pipeline(sig)
        
        if sig is None:
            logging.info(f"{files[i]} is invalid.")
            cnt+= 1
            continue   
        out_path = os.path.join(OUTPUT_DIR,files[i])
        os.makedirs(os.path.dirname(out_path),exist_ok=True)
        np.save(out_path,sig)
    logging.info(f"invalid files:{cnt}")
        
if __name__ == '__main__':
    main()