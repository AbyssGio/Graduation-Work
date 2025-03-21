import argparse

import os
import jittor as jt
from model.df_net import CDFNet  # Y ?
from train.df_trainer import DFTrainer  # Y ???
from utils.config_loader import CFG  # Y

os.environ["CUDA_VISIBLE_DEVICES"] = "1, 0"
jt.flags.use_cuda = 1

# np.random.seed(0)
# torch.manual_seed(0)
# torch.cuda.manual_seed(0)
# torch.cuda.manual_seed_all(0)
# random.seed(0)


def get_param_num(m):
    total = sum(p.numel() for p in m.parameters())
    print("{:.2f}".format(total / (1024 * 1024)))


def build_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("cfg_file", type=str, help="配置文件位置")


def main():
    cfg_file_path = "experiments/df_L50.yml"
    cfg = CFG.read_from_yaml(cfg_file_path)
    print("Load configuration from {}".format(cfg_file_path))
    if cfg.general.model == "DFNet":
        model = CDFNet(cfg)
        model_trainer = DFTrainer(cfg, model)
    else:
        raise Exception("Unknown model")

    if cfg.general.running_type == "train":
        model_trainer.start_training()
    elif cfg.general.running_type == "val":
        print(model_trainer.start_validating())


if __name__ == '__main__':
    main()
