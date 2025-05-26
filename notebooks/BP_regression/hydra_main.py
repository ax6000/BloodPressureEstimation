import hydra
from omegaconf import DictConfig,OmegaConf
import wandb
import os
from hydra_utils import torch_fix_seed,output_warning
from hydra.utils import instantiate,call
# from hydra_BPClassification import *
# モデルを変えて連続で実行する

# run selected function  
    
@hydra.main(config_name="config", version_base=None, config_path="conf")
def main(cfg: DictConfig) -> None:
    torch_fix_seed(42)
    wandb.config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    print(cfg)
    run = wandb.init(entity=cfg.other.wandb.entity, project=cfg.other.wandb.project)
    # Iterate over tasks without instantiating them
    call(cfg.task,_recursive_=False)
    wandb.finish()
if __name__ == "__main__":
    main()