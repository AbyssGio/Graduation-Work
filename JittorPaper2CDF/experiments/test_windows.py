from model.df_net import DFNet
import time
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch.autograd import Variable

from train.df_trainer import DFTrainer
from utils.config_loader import CFG
from utils.inverse_warp import inverse_warp_pytorch
from utils.normal2depth import normal_to_depth
from utils.rgbd2pcd import get_xyz
from utils.visualize import vis_depth


class DFNetV2(DFNet):

    def __init__(self, cfg):
        self.w = max(cfg.model.forward_windows, 1)
        super().__init__(cfg)
        # for p in self.parameters():
        #     p.requires_grad = False
        # for p in self.rgb_model.parameters():
        #     p.requires_grad = True
        # for p in self.cost_volume_header.parameters():
        #     p.requires_grad = True


    def forward(self, batch):
        b, _, h, w = batch["cur_color"].size()
        cur_img_feat = self.rgb_model(batch["cur_color"])

        b, c, hh, ww = cur_img_feat.size()
        device = cur_img_feat.device
        cur_mask = F.interpolate(batch["mask"], size=(hh, ww), mode="bilinear", align_corners=True)
        min_depth = batch["min_depth"][:, :, :hh, :ww]
        raw_depth = F.interpolate(batch["raw_depth"], size=(hh, ww), mode="bilinear", align_corners=True)

        depth_gap = (1 - min_depth) / self.L
        # depth = Variable(torch.ones(b, 1, hh, ww)).to(device) * min_depth

        total_cost = Variable(torch.zeros(b, 1, self.L, hh, ww)).to(device)
        # for l in range(self.L):
        #     depth = min_depth + depth_gap * l
        #     # predict locally
        #     # depth[~(cur_mask > 0)] = raw_depth[~(cur_mask > 0)]

        for i in range(1, self.w + 1):
            cost = Variable(torch.zeros(b, c * 2, self.L, hh, ww)).to(device)

            R_mat = batch["forward_{}_R_mat".format(i)]
            t_vec = batch["forward_{}_t_vec".format(i)]
            ref_img_feat = self.rgb_model(batch["forward_{}_color".format(i)])

            for l in range(self.L):
                depth = min_depth + depth_gap * l
                depth[~(cur_mask > 0)] = raw_depth[~(cur_mask > 0)]
                cost[:, :c, l, :, :] = cur_img_feat
                cost[:, c:, l, :, :] = inverse_warp_pytorch(depth, ref_img_feat, R_mat, t_vec,
                                                            batch["fx"], batch["fy"], batch["cx"], batch["cy"])

            cost = cost.contiguous()
            cost = self.cost_volume_header(cost)
            for n in range(self.cost_volume_layer_num):
                cost = cost + self.residual_cost_volume[n](cost)
            cost = self.cost_volume_tail(cost)
            total_cost = total_cost + cost

        total_cost = total_cost / self.w

        if self.add_pt_feature and self.refine_cost:
            refined_cost = Variable(torch.zeros(b, 1, self.L, hh, ww)).to(device)
            cur_pt_feature = self.pt_model(batch["pt"])
            for i in range(self.L):
                refined_cost[:, :, i, :, :] = self.context_network(
                    torch.cat([cur_img_feat, total_cost[:, :, i, :, :], cur_pt_feature], 1)
                ) + total_cost[:, :, i, :, :]
        else:
            if self.refine_cost:
                refined_cost = Variable(torch.zeros(b, 1, self.L, hh, ww)).to(device)
                for i in range(self.L):
                    refined_cost[:, :, i, :, :] = self.context_network(
                        torch.cat([cur_img_feat, total_cost[:, :, i, :, :]], 1)
                    ) + total_cost[:, :, i, :, :]
            else:
                refined_cost = total_cost

        refined_cost_squeezed = torch.squeeze(refined_cost, 1)
        refined_cost_squeezed = F.softmax(
            F.interpolate(refined_cost_squeezed, [h, w], mode='bilinear', align_corners=True), dim=1
        )
        pred = self.depth_regression(refined_cost_squeezed)
        depth = batch["min_depth"] + ((1 - batch["min_depth"]) / self.L) * pred.unsqueeze(1)
        depth = depth * batch["depth_scale"]
        return depth


if __name__ == '__main__':
    cfg_file_path = "./dfv2_2.yml"
    cfg = CFG.read_from_yaml(cfg_file_path)
    print("Load configuration from {}".format(cfg_file_path))
    if cfg.general.running_type == "train":
        model = DFNetV2(cfg)
        model_trainer = DFTrainer(cfg, model)
        model_trainer.start_training()

    elif cfg.general.running_type == "val":
        for w in range(10):
            cfg.model.forward_windows = w
            model = DFNetV2(cfg)
            model.eval()
            model_trainer = DFTrainer(cfg, model)
            with torch.no_grad():
                print(model_trainer.start_validating())
    elif cfg.general.running_type == "test":
        for w in range(10):
            cfg.model.forward_windows = w
            model = DFNetV2(cfg)
            model.eval()
            model_trainer = DFTrainer(cfg, model)
            for idx, batch in enumerate(model_trainer.val_data_loader):
                # if idx < 50:
                #     continue
                for k in batch:
                    batch[k] = batch[k].to(model_trainer.device)
                eval_time_start = time.time()
                with torch.no_grad():
                    output_depth = model(batch)

                    output_depth[~(batch["mask"] > 0)] = batch["raw_depth"][~(batch["mask"] > 0)]
                print(time.time() - eval_time_start)
                print(torch.cuda.memory_allocated())
                print(torch.cuda.max_memory_allocated())
                gt_depth = batch["cur_gt_depth"] / batch["depth_scale"]
                gt_depth = gt_depth.squeeze().cpu().numpy()
                output_depth = output_depth.squeeze().cpu().numpy()
                # np.abs(gt_depth-output_depth) # color_map="gray"
                # vis_depth(output_depth, visualize=True)
                break
    else:
        pass
