import os
import importlib
import numpy as np
import torch
from omegaconf import OmegaConf
import random

def torch_fix_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

def update_if_exists(conf, key, value):
    if OmegaConf.select(conf, key) is not None:
        OmegaConf.update(conf, key, value)

# from https://github.com/CompVis/latent-diffusion/blob/main/ldm/util.py
def instantiate_from_config(config, **kwargs):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    cls_or_fn = get_obj_from_str(config["target"])
    params = config.get("params", dict())
    
    # kwargs の値で params を上書きするようにマージ
    merged_params = {**params, **kwargs}
    
    return cls_or_fn(**merged_params)


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def output_warning(config):
    # 'output_path'が存在する場合
    if os.path.exists(config.network.ckpt_path):
        # ユーザーに確認を求める（enterを押したらyes）
        confirmation = input(f"Warning: The output path {config.network.ckpt_path} already exists. Do you want to continue? Press Enter to continue: ").strip().lower()
        
        # 入力が空（Enter）が押された場合は続行、それ以外はエラー
        if confirmation == "" or confirmation == "y":
            print("Proceeding with the operation...")
        else:
            raise ValueError("Operation aborted by the user.")
    else:
        print(f"The output path {config.network.ckpt_path} does not exist. Proceeding...")

def compute_class_weights(labels):
# クラスごとのサンプル数をカウント
    unique, counts = np.unique(labels, return_counts=True)
    total_samples = len(labels)
    
    # クラスの重みを計算 (サンプル数の逆数で重み付け)
    weights = total_samples / (len(unique) * counts)
    return torch.FloatTensor(weights)

class EarlyStopping:
    """earlystoppingクラス"""

    def __init__(self, path, patience=5, verbose=False):
        """引数：最小値の非更新数カウンタ、表示設定、モデル格納path"""

        self.patience = patience    #設定ストップカウンタ
        self.verbose = verbose      #表示の有無
        self.counter = 0            #現在のカウンタ値
        self.best_score = None      #ベストスコア
        self.early_stop = False     #ストップフラグ
        # self.val_loss_min = np.Inf   #前回のベストスコア記憶用
        
        self.path = path             #ベストモデル格納path
        os.makedirs(os.path.dirname(self.path),exist_ok=True)
    def __call__(self, val_loss, model):
        """
        特殊(call)メソッド
        実際に学習ループ内で最小lossを更新したか否かを計算させる部分
        """
        score = val_loss

        if self.best_score is None:  #1Epoch目の処理
            self.best_score = score
            self.checkpoint(score, model)  #記録後にモデルを保存してスコア表示する
        elif score > self.best_score:  # ベストスコアを更新できなかった場合
            self.counter += 1   #ストップカウンタを+1
            if self.verbose:  #表示を有効にした場合は経過を表示
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')  #現在のカウンタを表示する
                print(f"the best of loss: {self.best_score:.5f}")
            if self.counter >= self.patience:  #設定カウントを上回ったらストップフラグをTrueに変更
                self.early_stop = True
        else:  #ベストスコアを更新した場合
            self.checkpoint(score, model)  #モデルを保存してスコア表示
            self.counter = 0  #ストップカウンタリセット

    def checkpoint(self, score, model):
        '''ベストスコア更新時に実行されるチェックポイント関数'''
        if self.verbose:  #表示を有効にした場合は、前回のベストスコアからどれだけ更新したか？を表示
            print(f'Validation loss decreased ({self.best_score:.5f} --> {score:.5f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)  #ベストモデルを指定したpathに保存
        self.best_score = score #その時のlossを記録する