import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from einops import rearrange
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(pt, npoint):
    """
    Input:
        pt: pointcloud data, [B, 3, N]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    xyz = pt.transpose(2, 1)  # [B,3,N] -> [B,N,3]
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return index_points(xyz, centroids).transpose(2, 1)


class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 512, 1)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(256)
        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)

        x = F.relu(self.bn4(self.fc1(x)))
        x = self.fc2(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, feature_transform=False, out_channel=64):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(3)
        self.hidden_channel = 64
        self.conv1 = torch.nn.Conv1d(3, self.hidden_channel, 1)
        self.conv2 = torch.nn.Conv1d(self.hidden_channel, out_channel, 1)
        self.bn1 = nn.BatchNorm1d(self.hidden_channel)
        self.bn2 = nn.BatchNorm1d(out_channel)
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class PointNet(nn.Module):

    def __init__(self, img_size, out_channels):
        super().__init__()
        if not out_channels:
            self.out_channels = [512, 256, 128, 64]
        else:
            self.out_channels = out_channels
        self.stn = STN3d(3)
        self.hws = [
            [img_size[0] // 32, img_size[1] // 32],
            [img_size[0] // 16, img_size[1] // 16],
            [img_size[0] // 8, img_size[1] // 8],
            [img_size[0] // 4, img_size[1] // 4]
        ]
        self.num_samples = [hw[0] * hw[1] for hw in self.hws]
        self.pt_encoder_1 = PointNetEncoder(out_channel=self.out_channels[0], feature_transform=True)
        self.pt_encoder_2 = PointNetEncoder(out_channel=self.out_channels[1], feature_transform=True)
        self.pt_encoder_3 = PointNetEncoder(out_channel=self.out_channels[2], feature_transform=True)
        self.pt_encoder_4 = PointNetEncoder(out_channel=self.out_channels[3], feature_transform=True)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, pt):
        # pt: (b s=3 n=4096)
        trans = self.stn(pt)
        pt = pt.transpose(2, 1)
        pt = torch.bmm(pt, trans)
        pt = pt.transpose(2, 1)
        p3 = self.pt_encoder_4(pt)
        pt = farthest_point_sample(pt, self.num_samples[2])
        p2 = self.pt_encoder_3(pt)
        pt = farthest_point_sample(pt, self.num_samples[1])
        p1 = self.pt_encoder_2(pt)
        pt = farthest_point_sample(pt, self.num_samples[0])
        p0 = self.pt_encoder_1(pt)
        p0 = self.up(rearrange(p0, "b s (h w) -> b s h w", h=self.hws[0][0], w=self.hws[0][1]))
        p1 = self.up(rearrange(p1, "b s (h w) -> b s h w", h=self.hws[1][0], w=self.hws[1][1]))
        p2 = self.up(rearrange(p2, "b s (h w) -> b s h w", h=self.hws[2][0], w=self.hws[2][1]))
        p3 = self.up(rearrange(p3, "b s (h w) -> b s h w", h=self.hws[3][0], w=self.hws[3][1]))

        return p0, p1, p2, p3


class VPointNet(nn.Module):

    def __init__(self, img_size, hidden_dim=64, out_channels=16):
        super().__init__()
        self.h = img_size[0] // 4
        self.w = img_size[1] // 4
        self.num_samples = self.h * self.w
        self.pt_encoder_1 = PointNetEncoder(feature_transform=True, out_channel=hidden_dim)
        # self.pt_encoder_2 = PointNetEncoder(feature_transform=True, out_channel=hidden_dim // 2)
        # self.pt_encoder_2 = PointNetEncoder(feature_transform=True, out_channel=hidden_dim // 2)

        # self.concentration_dim = hidden_dim + hidden_dim // 2 + hidden_dim // 4
        # self.concentration_dim = hidden_dim + hidden_dim // 2
        self.concentration_dim = hidden_dim

        self.conv_1 = torch.nn.Conv1d(self.concentration_dim, out_channels, 1)
        self.bn_1 = nn.BatchNorm1d(out_channels)
        self.stn = STN3d(3)

    def forward(self, pt):
        trans = self.stn(pt)
        pt = pt.transpose(2, 1)
        pt = torch.bmm(pt, trans)
        pt = pt.transpose(2, 1)
        # p2 = self.pt_encoder_3(pt)
        # pt = farthest_point_sample(pt, self.num_samples)  # b, hidden_dim /2, h/4, w/4
        p2 = self.pt_encoder_1(pt)

        # pt = farthest_point_sample(pt, self.num_samples // 4)  # b, hidden_dim, h/8, w/8
        # p1 = self.pt_encoder_1(pt_sampled)

        # pt = farthest_point_sample(pt, self.num_samples // 16)
        # p0 = self.pt_encoder_2(pt)

        # p1 = self.up(rearrange(p1, "b s (h w) -> b s h w", h=self.h//2, w=self.w//2))
        # p2 = rearrange(p2, "b s h w -> b s (h w)",  h=self.h, w=self.w)
        # p2 = torch.cat([p1, p2], dim=1)
        x = F.relu(self.bn_1(self.conv_1(p2)))
        x = rearrange(x, "b s (h w) -> b s h w", h=self.h, w=self.w)
        return x
