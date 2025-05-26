import numpy as np
import pickle
from torch.utils.data import  Dataset
import torch

class BPDataset(Dataset):
    def __init__(self, data_dir,cv=False,fold=None,train=True,data_len=-1):
        # Load data
        if train:
            if cv:
                self.x = np.load(f'{data_dir}/train_2.npy')[:, 1, :].reshape(-1,1,1250)
                self.y = np.load(f'{data_dir}/train_sbp_2labels.npy')
                with open(f'{data_dir}/cv_5fold_2labels.pkl','rb') as f:
                    file = pickle.load(f)
                    self.x = self.x[file[0][fold]]
                    self.y = self.y[file[0][fold]]
                # with open(f'{data_dir}/ppgidx_2labels_cv.pkl','rb') as f:
                #     ppgidx_2labels_cv = pickle.load(f)
                # self.x = self.x[ppgidx_2labels_cv[0][fold]]
                # with open(f'{data_dir}/sbp_2labels_cv.pkl','rb') as f:
                #     sbp_4labels_cv = pickle.load(f)
            else:
                self.x = np.load(f'{data_dir}/train_2.npy')[:, 1, :].reshape(-1,1,1250)  # Shape: (-1, 1250)
                self.y = np.load(f'{data_dir}/train_sbp_2labels.npy')  # Shape: (-1,)
            print(self.x.shape,self.y.shape)
            # self.x = np.load(f'{data_dir}/train_2.npy')[:, 1, :].reshape(-1,1,1250)  # Shape: (-1, 1250)
            # self.y = np.load(f'{data_dir}/train_sbp_2labels.npy')  # Shape: (-1,)
        else:
            if cv:
                self.x = np.load(f'{data_dir}/train_2.npy')[:, 1, :].reshape(-1,1,1250) # Shape: (-1, 1250)
                self.y = np.load(f'{data_dir}/train_sbp_2labels.npy')
                with open(f'{data_dir}/cv_5fold_2labels.pkl','rb') as f:
                    file = pickle.load(f)
                    self.x = self.x[file[1][fold]]
                    self.y = self.y[file[1][fold]]
                # with open(f'{data_dir}/ppgidx_2labels_cv.pkl','rb') as f:
                #     ppgidx_2labels_cv = pickle.load(f)
                # self.x = self.x[ppgidx_2labels_cv[1][fold]]
                # with open(f'{data_dir}/sbp_2labels_cv.pkl','rb') as f:
                #     sbp_4labels_cv = pickle.load(f)
                # self.y = sbp_4labels_cv[1][fold]  # Shape: (-1,)
            else:
                self.x = np.load(f'{data_dir}/test_2.npy')[:, 1, :].reshape(-1,1,1250)  # Shape: (-1, 1250)
                self.y = np.load(f'{data_dir}/test_sbp_2labels.npy')  # Shape: (-1,)
            # self.x = np.load(f'{data_dir}/test_2.npy')[:, 1, :].reshape(-1,1,1250)  # Shape: (-1, 1250)
            # self.y = np.load(f'{data_dir}/test_sbp_2labels.npy')  # Shape: (-1,)
        # Convert to torch tensors
        if data_len != -1:
            self.x = self.x[:min(data_len,len(self.x))]
            self.y = self.y[:min(data_len,len(self.y))]
        self.x = torch.FloatTensor(self.x)
        self.y = torch.LongTensor(self.y)
        print(self.x.shape,self.y.shape)
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
class BPDataset_Regr(Dataset):
    def __init__(self, data_dir,cv=None,train=True):
        # Load data
        if train:
            self.x = np.load(f'{data_dir}/train.npy')[:, 1, :].reshape(-1,1,1250)  # Shape: (-1, 1250)
            self.sbp = np.load(f'{data_dir}/train_sbp.npy')  # Shape: (-1,)
            self.dbp = np.load(f'{data_dir}/train_dbp.npy')  # Shape: (-1,)
        else:
            if cv is not None:
                self.x = np.load(f'{data_dir}/train.npy')[:, 1, :].reshape(-1,1,1250)  # Shape: (-1, 1250)
                self.sbp = np.load(f'{data_dir}/train_sbp.npy')  # Shape: (-1,)
                self.dbp = np.load(f'{data_dir}/train_dbp.npy')

            else:
                self.x = np.load(f'{data_dir}/test.npy')[:, 1, :].reshape(-1,1,1250)  # Shape: (-1, 1250)
                self.sbp = np.load(f'{data_dir}/test_sbp.npy')  # Shape: (-1,)
                self.dbp = np.load(f'{data_dir}/test_dbp.npy')  # Shape: (-1,)
        if cv is not None:
            self.x = self.x[cv]
            self.sbp = self.sbp[cv]
            self.dbp = self.dbp[cv]
        scale = np.load(f'{data_dir}/scale_train.npy')
        self.sbp = self.sbp * scale[0,1]- scale[0,0]
        self.dbp = self.dbp * scale[0,1] - scale[0,0]

        # Convert to torch tensors
        self.x = torch.FloatTensor(self.x)
        self.sbp = torch.FloatTensor(self.sbp)
        self.dbp = torch.FloatTensor(self.dbp)
        print(self.x.shape,self.sbp.shape,self.dbp.shape)
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.sbp[idx],self.dbp[idx]
