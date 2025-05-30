import jittor as jt
import jittor.nn as nn


def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_planes))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU())

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def execute(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out


class matchshifted(nn.Module):
    def __init__(self):
        super(matchshifted, self).__init__()

    def execute(self, left, right, shift):
        batch, filters, height, width = left.size()
        shifted_left = nn.pad(
            jt.index_select(left, 3, jt.array64([i for i in range(shift, width)]).cuda()),
            (shift, 0, 0, 0))
        shifted_right = nn.pad(
            jt.index_select(right, 3, jt.array64([i for i in range(width - shift)]).cuda()),
            (shift, 0, 0, 0))
        out = jt.concat((shifted_left, shifted_right), 1).view(batch, filters * 2, 1, height, width)
        return out



class PyramidPool32(nn.Module):
    def __init__(self, out_channel=32, hidden_dim=16):
        super(PyramidPool32, self).__init__()
        self.inplanes = hidden_dim
        self.out_channel = out_channel
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU())

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)

        self.branch1 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32, 32)),
                                     convbn(64, 32, 1, 1, 0, 1),
                                     nn.ReLU())

        self.branch2 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16, 16)),
                                     convbn(64, 32, 1, 1, 0, 1),
                                     nn.ReLU())

        self.branch3 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8, 8)),
                                     convbn(64, 32, 1, 1, 0, 1),
                                     nn.ReLU())

        self.branch4 = nn.Sequential(nn.AvgPool2d((4, 4), stride=(4, 4)),
                                     convbn(64, 32, 1, 1, 0, 1),
                                     nn.ReLU())

        self.lastconv = nn.Sequential(convbn(32*4, 32*2, 3, 1, 1, 1),
                                      nn.ReLU(),
                                      nn.Conv2d(32*2, self.out_channel, kernel_size=1, padding=0, stride=1, bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def execute(self, x):
        output = self.firstconv(x)
        output = self.layer1(output)
        output_raw = self.layer2(output)
        output = self.layer3(output_raw)
        output_skip = self.layer4(output)

        output_branch1 = self.branch1(output_skip)
        output_branch1 = nn.interpolate(output_branch1, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear',
                                       align_corners=False)

        output_branch2 = self.branch2(output_skip)
        output_branch2 = nn.interpolate(output_branch2, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear',
                                       align_corners=False)

        output_branch3 = self.branch3(output_skip)
        output_branch3 = nn.interpolate(output_branch3, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear',
                                       align_corners=False)

        output_branch4 = self.branch4(output_skip)
        output_branch4 = nn.interpolate(output_branch4, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear',
                                       align_corners=False)

        output_feature = jt.concat(
            (output_branch4, output_branch3, output_branch2, output_branch1), 1)
        output_feature = self.lastconv(output_feature)
        return output_feature


class PyramidPool(nn.Module):
    def __init__(self, out_channel=32, hidden_dim=32):
        super(PyramidPool, self).__init__()
        self.inplanes = hidden_dim
        self.out_channel = out_channel
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU())

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)

        self.branch1 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32, 32)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU())

        self.branch2 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16, 16)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU())

        self.branch3 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8, 8)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU())

        self.branch4 = nn.Sequential(nn.AvgPool2d((4, 4), stride=(4, 4)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU())

        self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                      nn.ReLU(),
                                      nn.Conv2d(128, self.out_channel, kernel_size=1, padding=0, stride=1, bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def execute(self, x):
        output = self.firstconv(x)
        output = self.layer1(output)
        output_raw = self.layer2(output)
        output = self.layer3(output_raw)
        output_skip = self.layer4(output)

        output_branch1 = self.branch1(output_skip)
        output_branch1 = nn.interpolate(output_branch1, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear',
                                       align_corners=False)

        output_branch2 = self.branch2(output_skip)
        output_branch2 = nn.interpolate(output_branch2, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear',
                                       align_corners=False)

        output_branch3 = self.branch3(output_skip)
        output_branch3 = nn.interpolate(output_branch3, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear',
                                       align_corners=False)

        output_branch4 = self.branch4(output_skip)
        output_branch4 = nn.interpolate(output_branch4, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear',
                                       align_corners=False)

        output_feature = jt.concat(
            (output_raw, output_skip, output_branch4, output_branch3, output_branch2, output_branch1), 1)
        output_feature = self.lastconv(output_feature)
        return output_feature
