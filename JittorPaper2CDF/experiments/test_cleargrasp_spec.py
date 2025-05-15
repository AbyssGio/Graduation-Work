import argparse

import torch
import os

from torch.utils.data import DataLoader

from cfg.datasets.cleargrasp_dataset import ClearGrasp
from train.trainer import Trainer
from model.vdcnet import SwinVDC
from utils.config_loader import CFG

cfg_file_path = "../experiments/swin_pt_spade_config.yml"
cfg = CFG.read_from_yaml(cfg_file_path)
print("Load configuration from {}".format(cfg_file_path))
if cfg.general.model == "SwinVDC":
    model = SwinVDC(cfg)
else:
    raise Exception("Unknown model")
# model_trainer = Trainer(cfg, model)
model_trainer = Trainer(cfg, model)
h, w = cfg.general.frame_h, cfg.general.frame_w
test_dataset = {
    "real_known": ClearGrasp(cfg.dataset.dataset_dir, "val", h, w, specific_ds="real-val"),
    "real_novel": ClearGrasp(cfg.dataset.dataset_dir, "test", h, w, specific_ds="real-test"),
    "syn_known": ClearGrasp(cfg.dataset.dataset_dir, "val", h, w, specific_ds="synthetic-val"),
    "syn_novel": ClearGrasp(cfg.dataset.dataset_dir, "test", h, w, specific_ds="synthetic-test"),
}
for k, v in test_dataset.items():
    model_trainer.val_data_loader = DataLoader(
        test_dataset[k],
        batch_size=cfg.training.valid_batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers
    )
    with torch.no_grad():
        losses = model_trainer.start_validating()
        print(k)
        print(losses)
