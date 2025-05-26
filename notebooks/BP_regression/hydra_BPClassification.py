from se_resnet1d import resnet18 as resnet
from transformers import get_cosine_schedule_with_warmup
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import wandb
from sklearn.metrics import f1_score
# from resnet1d import ResNet1D
import pickle
from hydra_utils import output_warning,compute_class_weights,EarlyStopping,update_if_exists
from hydra.utils import instantiate,call
from sklearn.metrics import confusion_matrix
import seaborn as sns


class BP_Logger(object):
    # 共通のロガー？
    # メトリクスをコンフィグで指定して、関数の入力は固定にする？
    # 何をログするか引数（config）で指定する？
    def __init__(self):
        pass
    def log(self):
        pass
        
    def log_img(self):
        pass
    def log_metrics(self):  
        pass
    def create_figure(self,gt,pred):
        fig,ax = plt.subplots()
        ax.plot(gt,label="true")  
        ax.plot(gt,label="pred")
        ax.legend()
        return fig
    def log_img(self,gt,pred):
        gt, pred = gt[0].detach().clone().cpu().numpy(), pred[0].detach().clone().cpu().numpy()
        figure = self.create_figure(gt,pred)
        img = wandb.Image(figure)
        plt.clf()
        plt.close()
        return img   
    # モデルの評価指標
    def calculate_metrics(y_true, y_pred,classes=2):
        # 正解率
        accuracy = np.mean(y_true == y_pred)
        
        # クラスごとの精度を計算
        class_accuracies = []
        for cls in range(classes):
            mask = (y_true == cls)
            if np.sum(mask) > 0:
                class_acc = np.sum((y_true == y_pred) & mask) / np.sum(mask)
                class_accuracies.append(class_acc)
            else:
                raise ValueError(f"Class {cls} has no samples.")
        
        # マクロ平均F1スコア
        f1_macro = f1_score(y_true, y_pred, average='macro')
        
        return accuracy, class_accuracies, f1_macro 
class BP_Trainer(object):
    def __init__(self, config):
        self.config = config
        print(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = instantiate(self.config.network.logger)
        self._criterion = self.config.network.criterion
        self._early_stopping = self.config.network.early_stopping
        self._model = self.config.model
        self._optimizer = self.config.network.optimizer
        self._scheduler = self.config.network.scheduler
        self._data = self.config.data
        self.resume = self.config.network.resume
        self.fold = self.config.network.fold
        self.batch_size = self.config.network.batch_size
        self.epochs = self.config.network.epochs
        self.batch_size = self.config.network.batch_size
        self.log_interval = self.config.network.log_interval
        self.min_epochs = self.config.network.min_epochs
        self.output_path = self.config.network.ckpt_path
        # self.init_weights=init_weights
    def train(self):
        output_warning(self.config)
        data_dir = '../../data/processed/BP_npy/PulseDB'
        for f in range(self.fold):
            fold_now = f
            train_dataset = instantiate(self._data.train_dataset,fold=fold_now,train=True)
            val_dataset = instantiate(self._data.val_dataset,fold=fold_now,train=False)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
            unique, counts = np.unique(train_dataset.y, return_counts=True)
            for label, count in zip(unique, counts):
                print(f"ラベル {label}: {count}件")
            unique, counts = np.unique(val_dataset.y, return_counts=True)
            for label, count in zip(unique, counts):
                print(f"ラベル {label}: {count}件")
            # weight 
            labels = np.load(f'{data_dir}/train_sbp_2labels.npy')
            with open(f"{data_dir}/cv_5fold_2labels.pkl", "rb") as file:
                cv_idx = pickle.load(file) 
                labels = labels[cv_idx[0][f]]
            class_weights = compute_class_weights(labels).to(self.device)
            print(class_weights)

            self.model = instantiate(self._model)
            self.optimizer = instantiate(self._optimizer,params=self.model.parameters())
            update_if_exists(self._scheduler,"num_training_steps",self.min_epochs * len(train_loader))
            update_if_exists(self._scheduler,"num_warmup_steps",self._scheduler.num_training_steps * 0.1)
            # update_if_exists(self._criterion,"weight",class_weights.to(self.device))
            update_if_exists(self._early_stopping,"path",f"{self.output_path}/best_fold{f}.pth")
            self.scheduler = instantiate(self._scheduler,optimizer=self.optimizer)
            self.criterion = instantiate(self._criterion,weight=class_weights)
            self.earlystopping = instantiate(self._early_stopping)
            self.model.to(self.device)

            wandb.watch(self.model, log_freq=self.log_interval)
            for epoch in range(self.epochs):
                self.model.train()
                running_loss = 0.0
                # running_loss_mae = 0.0
                # Training phase with progress bar
                train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.epochs} - Training", leave=False)
                for batch_idx, (x,y) in enumerate(train_loader_tqdm):
                    x,y = x.to(self.device), y.to(self.device)
                    # print(gt.shape,cond.shape)
                    self.optimizer.zero_grad()
                    outputs = self.model(x)
                    loss = self.criterion(outputs, y) 
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    # self.logger.log()
                    if batch_idx % self.log_interval == 0:
                        wandb.log({"train_loss_step": loss.item(),
                                "lr": self.scheduler.get_last_lr()[0]})
                    # if batch_idx  == 0:
                        # wandb.log({"train/loss": log_img(gt,outputs)})
                    running_loss += loss.item()
                    # running_loss_mae += loss_mae.item()
                    train_loader_tqdm.set_postfix(loss=running_loss/(batch_idx+1))

                train_loss = running_loss / len(train_loader)
                # train_loss_mae = running_loss_mae / len(train_loader)

                # Validation phase with progress bar
                self.model.eval()
                val_loss = 0.0
                all_y_true = []
                all_y_pred = []
                val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{self.epochs} - Validation", leave=False)
                with torch.no_grad():
                    for batch_idx, (x,y) in enumerate(val_loader_tqdm):

                        x,y = x.to(self.device), y.to(self.device)
                        outputs = self.model(x)
                        # print(outputs.device,y.device)
                        loss = self.criterion(outputs,y) 
                        # loss_mae = mae(outputs, gt)
                        if batch_idx % self.log_interval == 0:
                            wandb.log({"test_loss_step": loss.item()})
                        # if batch_idx  == 0:
                        #     wandb.log({"val/loss": log_img(gt,outputs)})
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        # val_loss_mae += loss_mae.item()
                        all_y_true.extend(np.array(y.cpu().numpy()))
                        all_y_pred.extend(np.array(predicted.cpu().numpy()))
                all_y_true = np.array(all_y_true)
                all_y_pred = np.array(all_y_pred)
                accuracy, class_accuracies, f1 = calculate_metrics(all_y_true, all_y_pred,classes=self.config.model.num_classes)
                val_loss = val_loss / len(val_loader)
                # WandBでのログ記録例
                wandb.log({
                    "train_loss": train_loss,
                    "test_loss": val_loss,
                    "accuracy": accuracy.item(),
                    "f1_macro": f1,
                    "conf_mat": wandb.plot.confusion_matrix(y_true=all_y_true, preds=all_y_pred, class_names=["0","1"]),
                    "epoch": epoch + 1
                })
                # val_loss /= len(val_loader)
                # val_loss_mae /= len(val_loader)
                self.earlystopping(val_loss,self.model)
                if self.earlystopping.early_stop:
                    print("Early Stopping!")
                    break
                print(f"Epoch [{epoch + 1}/{self.epochs}]"
                    f" Train Loss: {train_loss:.4f}"
                    f" Val Loss: {val_loss:.4f}")
    def test(self):
        self.model = instantiate(self._model)
        checkpoint_dir =self.output_path
        f1s = []
        test_dataset = instantiate(self._data.test_dataset,train=False)
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        for fold in range(self.fold):
            # データセットとDataLoaderの作成
            # load checkpoint
            best_checkpoint_path = os.path.join(checkpoint_dir, f'best_fold{fold}.pth')
            self.model.load_state_dict(torch.load(best_checkpoint_path))
            with torch.no_grad():
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model.to(device)
                self.model.eval()

                all_y_true = []
                all_y_pred = []
                all_y_pred_proba = []
                # 進捗バーの設定
                val_loader_tqdm = tqdm(test_dataloader, desc=f"Fold {fold+1}/{5} - Test", leave=False)
                for batch_idx, (x,y) in enumerate(val_loader_tqdm):
                    x,y = x.to(device), y.to(device)
                    outputs = self.model(x)
                    _, predicted = torch.max(outputs.data, 1)
                    all_y_pred_proba.extend(outputs.cpu().numpy())
                    # 予測確率を取得
                    all_y_true.extend(y.cpu().numpy())
                    all_y_pred.extend(predicted.cpu().numpy())
                all_y_true = np.array(all_y_true)
                all_y_pred = np.array(all_y_pred)
                all_y_pred_proba = np.array(all_y_pred_proba)
            # 予測結果の保存
            np.save(f'{checkpoint_dir}\\predictions_fold{fold}.npy', all_y_pred)
            # 正解ラベルの保存
            np.save(f'{checkpoint_dir}\\true_labels_fold{fold}.npy', all_y_true)
            # 予測確率の保存
            np.save(f'{checkpoint_dir}\\predictions_proba_fold{fold}.npy', all_y_pred_proba)
            # calculate metrics
            accuracy, class_accuracies, f1 = calculate_metrics(all_y_true, all_y_pred,classes=self.config.model.num_classes)
            print(f"Fold {fold+1} - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
            f1s.append(f1)
            # plot confusion matrix
            conf_mat = confusion_matrix(all_y_true, all_y_pred, normalize='true')
            plt.figure(figsize=(6, 5))
            sns.heatmap(conf_mat, cmap = 'Blues', annot=True)
            plt.title(f'Fold {fold+1} - Confusion Matrix')
            plt.savefig(f'{checkpoint_dir}\\confusion_matrix_fold{fold}.png')
            plt.close()


# モデルの評価指標
def calculate_metrics(y_true, y_pred,classes=2):
    # 正解率
    accuracy = np.mean(y_true == y_pred)
    
    # クラスごとの精度を計算
    class_accuracies = []
    for cls in range(classes):
        mask = (y_true == cls)
        if np.sum(mask) > 0:
            class_acc = np.sum((y_true == y_pred) & mask) / np.sum(mask)
            class_accuracies.append(class_acc)
        else:
            raise ValueError(f"Class {cls} has no samples.")
    
    # マクロ平均F1スコア
    f1_macro = f1_score(y_true, y_pred, average='macro')
    
    return accuracy, class_accuracies, f1_macro





def call_train(trainer:BP_Trainer):
    trainer.train()
def call_test(trainer:BP_Trainer):
    trainer.test()    
def execute_tasks(trainer, tasks: list):
    print(tasks)
    trainer_instance = instantiate(trainer, _recursive_=False)
    print("####instantiate done####")
    for task in tasks:  # Ensure trainer is assigned to the task
        print("#############")
        print(task)
        print(trainer)
        print(task["trainer"])
        call(task,trainer = trainer_instance)
