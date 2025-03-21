import random

import jittor as jt
import jittor.nn as nn
import numpy as np
from einops import einops
import jittor.misc as jm
import outerjittor.Jcdist as jd
import outerjittor.maskselect as ms


from outerjittor.chamfer_dist import ChamferDistance
from outerjittor.cossim import cosine_similarity
from utils.rgbd2pcd import get_surface_normal_from_depth, get_xyz


class Metrics:

    def __init__(self, eps=1e-8):
        self.eps = eps

    def depth_rmse(self, pred, gt):
        diff = (pred - gt) ** 2
        diff = jt.sqrt(jt.sum(jt.array(diff), jt.array([1, 2, 3])))
        diff_mask = jt.isfinite(diff)
        return jt.mean(diff[diff_mask]).data()

    def masked_depth_rmse(self, pred, gt, mask):
        diff = (pred - gt) ** 2
        ele_num = jt.sum(jt.array(mask), jt.array([1, 2, 3])).data()
        diff = jt.sqrt(
            jt.sum(jt.array(diff * mask), jt.array([1, 2, 3]) / (ele_num + self.eps)))
        return jt.mean(diff).data()

    def depth_mae(self, pred, gt):
        diff = jt.abs(pred - gt)
        return jt.mean(diff).data()

    def masked_depth_mae(self, pred, gt, mask):
        diff = jt.abs(pred - gt).data()
        ele_num = jt.sum(jt.array(mask), jt.array([1, 2, 3])).data()
        diff = jt.sum(jt.array(diff * mask),jt.array( [1, 2, 3]) / (ele_num + self.eps)).data()
        return jt.mean(diff).data()

    def depth_rel(self, pred, gt):
        err = jt.abs(gt - pred).data() / (gt + self.eps)
        err[jt.isinf(err)] = 0
        err[jt.isnan(err)] = 0
        return jt.mean(err).data()

    def masked_depth_rel(self, pred, gt, mask):
        err = jt.abs(gt - pred).data() / (gt + self.eps)
        err = err * mask
        err = jt.sum(err, dims=(1, 2, 3)).item() / jt.sum(mask, dims=(1, 2, 3)).item()
        # ===============
        err[jt.isnan(err)] = 0
        # ===============
        err_mask = jt.isfinite(err)
        return jt.mean(err[err_mask])

    def _depth_failing(self, pred, gt, threshold):
        err = gt / (pred + self.eps)
        err_reverse = pred / (gt + self.eps)
        err = jt.max(err, err_reverse)
        err_mask = err < threshold
        err[~err_mask] = 0
        count = 0
        for item in err.data():
            if item != 0:
                count = count + 1
        return count / err.numel()

    def _masked_depth_failing(self, pred, gt, mask, threshold):
        err = gt / (pred + self.eps)
        err_reverse = pred / (gt + self.eps)
        err = jt.max(err, err_reverse)
        err = jt.where(err > threshold, 0, 1)
        err = err * mask
        err = (jt.sum(err, dims=(1, 2, 3)).item() / jt.sum(mask, dims=(1, 2, 3)).item())
        # ===============
        err[jt.isnan(err)] = 0
        # ===============
        err_mask = jt.isfinite(err)
        return jt.mean(err[err_mask]).item()

    def depth_failing_105(self, pred, gt, mask):
        return self._masked_depth_failing(pred, gt, mask, 1.05)

    def depth_failing_110(self, pred, gt, mask):
        return self._masked_depth_failing(pred, gt, mask, 1.10)

    def depth_failing_125(self, pred, gt, mask):
        return self._masked_depth_failing(pred, gt, mask, 1.25)

    def depth_failing_115(self, pred, gt, mask):
        return self._masked_depth_failing(pred, gt, mask, 1.15)


def depth_loss(pred, gt, mask, beta=1.0, reduction="sum"):
    b = pred.size()[0]
    pred_copy = pred * mask
    gt_copy = gt * mask
    if reduction == "mean":
        return nn.smooth_l1_loss(pred_copy, gt_copy, reduction="sum") / (jt.sum(mask).item() + 1e-8)
        # return F.smooth_l1_loss(pred_copy, gt_copy, reduction="mean", beta=beta) ?
    else:
        return nn.smooth_l1_loss(pred_copy, gt_copy, reduction="mean")


def weighted_depth_loss(pred, gt, mask, beta=1.0, reduction="sum", weight=0.8):
    b = pred.size()[0]
    weighted_mask = jt.where(mask > 0, weight, 1-weight)
    pred_copy = pred * weighted_mask
    gt_copy = gt * weighted_mask
    if reduction == "mean":
        return nn.smooth_l1_loss(pred_copy, gt_copy, reduction="sum") / (jt.sum(mask).item() + 1e-8)
        # return F.smooth_l1_loss(pred_copy, gt_copy, reduction="mean", beta=beta)
    else:
        return nn.smooth_l1_loss(pred_copy, gt_copy, reduction="sum") / b

def pairwise_L1_depth_loss(pred, gt, mask, eps=1e-6):
    b = pred.size()[0]
    res = 0
    for i in range(b):
        b_mask = mask[i, 0, :, :] > 0
        gt_flatten = ms.masked_select(gt[i, 0, :, :], b_mask)
        pred_flatten = ms.masked_select(pred[i, 0, :, :], b_mask)
        diff = jt.abs(jt.log(pred_flatten / (gt_flatten + eps)))
        diff2 = jt.abs(jt.log(gt_flatten / (pred_flatten + eps)))
        diff[jt.isnan(diff)] = 0.0
        diff[jt.isinf(diff)] = 0.0
        diff2[jt.isinf(diff2)] = 0.0
        diff2[jt.isnan(diff2)] = 0.0
        d = diff.size()[0]
        index = jt.array64(random.sample(
            range(d), min(4000, d)))
        diff = diff.view(-1, 1)
        diff2 = diff2.view(-1, 1)
        diff = jm.index_select(diff, 0, index)
        diff2 = jm.index_select(diff2, 0, index)
        # p=1 Manhattan Distance; p=2 Eud
        res += (jt.sum(jd.cdist(diff, diff2)).item() / (d * d))
    return res / b


class SSIMLoss(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(SSIMLoss, self).__init__()
        k = 7
        self.mu_x_pool = nn.AvgPool2d(k, 1)
        self.mu_y_pool = nn.AvgPool2d(k, 1)
        self.sig_x_pool = nn.AvgPool2d(k, 1)
        self.sig_y_pool = nn.AvgPool2d(k, 1)
        self.sig_xy_pool = nn.AvgPool2d(k, 1)

        self.refl = nn.ReflectionPad2d(k // 2)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * \
                 (sigma_x + sigma_y + self.C2)

        return jt.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def surface_normal_cos_loss(pred, gt_surface_normal, mask, camera, reduction="sum"):
    b = pred.size()[0]
    b_mask = mask.squeeze(1)  # squeeze with specific dim
    pred_sn = get_surface_normal_from_depth(
        pred, camera["fx"], camera["fy"], camera["cx"], camera["cy"])
    sn_loss = 1 - cosine_similarity(pred_sn, gt_surface_normal, dim=1)
    sn_loss = sn_loss[b_mask > 0]
    if reduction == "mean":
        return jt.mean(sn_loss).item()
    else:
        return jt.sum(sn_loss).item() / b


def surface_normal_l1_loss(pred_sn, gt_sn, mask):
    sn_mask = mask.repeat(1, 3, 1, 1)
    # sn_mask = torch.where(sn_mask > 0, 0.95, 0.05)
    if mask.sum() > 0:
        sn_loss = nn.smooth_l1_loss(pred_sn*sn_mask, gt_sn*sn_mask)
    else:
        sn_loss = 0
    return sn_loss


def surface_normal_loss(nmap, gt_surface_normal, mask):
    # b = nmap.size()[0]
    b_mask = mask.squeeze(1)
    b_mask = np.where(b_mask > 0, 0.8, 0.2)
    sn_loss = 1 - cosine_similarity(nmap, gt_surface_normal, dim=1)
    sn_loss = sn_loss * b_mask
    return jt.mean(sn_loss).item()


class PtGriddingLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.chamfer_distance = ChamferDistance()

    def forward(self, pred, gt_xyz, mask, camera):
        b = pred.size()[0]
        res = 0.
        pred_xyz = get_xyz(
            pred, camera["fx"], camera["fy"], camera["cx"], camera["cy"]).float()
        for i in range(b):
            b_mask = mask[i, 0, :, :] > 0
            obj_xyz_pred = pred_xyz[i, :, :, :].permute(1, 2, 0)
            obj_xyz_gt = gt_xyz[i, :, :, :].permute(1, 2, 0)
            obj_xyz_gt = obj_xyz_gt[b_mask].unsqueeze(0)
            obj_xyz_pred = obj_xyz_pred[b_mask].unsqueeze(0)
            # print(torch.isnan(obj_xyz_gt).any())
            # print(torch.isnan(obj_xyz_pred).any())
            # print(obj_xyz_gt.size(), obj_xyz_pred.size())
            dist1, dist2 = self.chamfer_distance(obj_xyz_pred, obj_xyz_gt)
            res += (jt.sum(dist1) + jt.sum(dist2))
        return res / b


class ConsLoss(nn.Module):
    def __init__(self, sobel_kernel=None, reduction="sum"):
        super(ConsLoss, self).__init__()
        if sobel_kernel is None:
            edge_kernel_x = jt.array(
                np.array([[1 / 8, 0, -1 / 8], [1 / 4, 0, -1 / 4], [1 / 8, 0, -1 / 8]]))
            edge_kernel_y = jt.array(
                np.array([[1 / 8, 1 / 4, 1 / 8], [0, 0, 0], [-1 / 8, -1 / 4, -1 / 8]]))
            self.sobel_kernel = jt.concat(
                (edge_kernel_x.view(1, 1, 3, 3), edge_kernel_y.view(1, 1, 3, 3)), dim=0)
            self.sobel_kernel.requires_grad = False
        else:
            self.sobel_kernel = sobel_kernel
        self.type_init = False
        self.reduction = reduction

    def get_grad_1(self, depth):
        if not self.type_init:
            self.sobel_kernel = self.sobel_kernel.type_as(depth)
        grad_depth = nn.conv2d(depth, self.sobel_kernel, padding=1)

        return -1 * grad_depth

    def get_grad_2(self, depth, nmap, fx, fy, cx, cy):
        p_b, _, p_h, p_w = depth.size()
        fx = einops.repeat(fx, 'bs -> bs h w', h=p_h, w=p_w)
        fy = einops.repeat(fy, 'bs -> bs h w', h=p_h, w=p_w)
        cy = einops.repeat(cy, 'bs -> bs h w', h=p_h, w=p_w)
        cx = einops.repeat(cx, 'bs -> bs h w', h=p_h, w=p_w)
        p_y = jt.arange(0, p_h).view(1, p_h, 1).expand(p_b, p_h, p_w).type_as(depth) - cy  # v - v_c
        p_x = jt.arange(0, p_w).view(1, 1, p_w).expand(p_b, p_h, p_w).type_as(depth) - cx  # u - u_c

        nmap_z = nmap[:, 2, :, :]
        nmap_z_mask = (nmap_z == 0)
        nmap_z[nmap_z_mask] = 1e-9
        nmap[:, 2, :, :] = nmap_z
        n_grad = nmap[:, :2, :, :].clone()
        n_grad = n_grad / (nmap[:, 2, :, :].unsqueeze(1))  # n_x / n_z, n_y / n_z

        grad_depth = -n_grad * depth

        f = jt.stack((fx, fy), dim=1)
        grad_depth = grad_depth / f

        denominator = 1
        denominator += p_x * (n_grad[:, 0, :, :]) / fx
        denominator += p_y * (n_grad[:, 1, :, :]) / fy
        denominator[denominator == 0] = 1e-9
        grad_depth = grad_depth / denominator.unsqueeze(1)

        return grad_depth

    def forward(self, depth, nmap, mask, gt_depth, camera):
        batch_size = depth.size()[0]
        _mask = mask > 0
        g_mask = _mask.expand(-1, 2, -1, -1)

        predict_nmap = get_surface_normal_from_depth(depth, camera["fx"], camera["fy"], camera["cx"], camera["cy"])

        true_grad_depth_1 = self.get_grad_1(gt_depth) * 100
        grad_depth_1 = self.get_grad_1(depth) * 100

        true_grad_depth_2 = self.get_grad_2(gt_depth, nmap, camera["fx"], camera["fy"], camera["cx"],
                                            camera["cy"]) * 100
        grad_depth_2 = self.get_grad_2(depth, predict_nmap, camera["fx"], camera["fy"], camera["cx"],
                                       camera["cy"]) * 100

        g_mask = (abs(true_grad_depth_1) < 1).type_as(g_mask) & (g_mask)
        g_mask = (abs(grad_depth_1) < 5).type_as(g_mask) & (
                abs(grad_depth_2) < 5).type_as(g_mask) & (g_mask)
        g_mask.detach_()
        if self.reduction == "sum":
            out1 = nn.smooth_l1_loss(grad_depth_1[g_mask], true_grad_depth_1[g_mask], reduction="sum") / batch_size
            out2 = nn.smooth_l1_loss(grad_depth_2[g_mask], true_grad_depth_2[g_mask], reduction="sum") / batch_size
        else:
            out1 = nn.smooth_l1_loss(grad_depth_1[g_mask], true_grad_depth_1[g_mask])
            out2 = nn.smooth_l1_loss(grad_depth_2[g_mask], true_grad_depth_2[g_mask])
        return out1 + out2


def depth_grad_loss_unsupervised(disp, img, mask):
    """
    Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    # normalize
    mean_disp = disp.mean(2, True).mean(3, True)
    norm_disp = disp / (mean_disp + 1e-7)
    disp = norm_disp
    grad_disp_x = jt.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = jt.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])
    grad_img_x = jt.mean(
        jt.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdims=True)
    grad_img_y = jt.mean(
        jt.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdims=True)

    # b x c x h(y) x w(x)
    grad_disp_x *= jt.exp(grad_img_x.negative())
    grad_disp_y *= jt.exp(grad_img_y.negative())
    mask_x = mask[:, :, :, :-1]
    mask_y = mask[:, :, :-1, :]
    loss = grad_disp_x[mask_x > 0].mean() + grad_disp_y[mask_y > 0].mean()
    return loss


def depth_grad_loss(pred, gt, mask):
    grad_pred_x = jt.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
    grad_pred_y = jt.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
    grad_gt_x = jt.abs(gt[:, :, :, :-1] - gt[:, :, :, 1:])
    grad_gt_y = jt.abs(gt[:, :, :-1, :] - gt[:, :, 1:, :])
    mask_x = mask[:, :, :, :-1]
    mask_y = mask[:, :, :-1, :]
    loss_x = nn.smooth_l1_loss(grad_pred_x[mask_x > 0], grad_gt_x[mask_x > 0])
    loss_y = nn.smooth_l1_loss(grad_pred_y[mask_y > 0], grad_gt_y[mask_y > 0])
    return loss_x + loss_y
