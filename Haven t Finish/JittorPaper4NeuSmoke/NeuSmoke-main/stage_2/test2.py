from __future__ import print_function
import argparse
import os

import jittor as jt
import jittor.nn as nn

import time

from dataloader import Smoke_data2 as lt
from dataloader import Smoke_loader2 as DA
from models import *
import jittor.transform as transforms


parser = argparse.ArgumentParser(description='SRNet')
parser.add_argument('--model', default='',
                    help='select model')
parser.add_argument('--datapath', default='',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=50,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default= None,
                    help='load model')
parser.add_argument('--savemodel', default='/root/siton-gpfs-archive/qiujiaxiong/data_smoke/s2n/results_real2_fea_rgbd_gradn2_noar/novel',
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
# args.cuda = not args.no_cuda and torch.cuda.is_available()

os.makedirs(args.savemodel, exist_ok=True)
os.makedirs(args.savemodel+'/att', exist_ok=True)
os.makedirs(args.savemodel+'/res', exist_ok=True)
#
# torch.manual_seed(args.seed)
# if args.cuda:
#     torch.cuda.manual_seed(args.seed)
jt.set_global_seed(args.seed)

datapath1 = '/root/siton-gpfs-archive/qiujiaxiong/data_smoke/s2n/data/real/test2'
all_left_img,  train_depth = lt.dataloader(datapath1)

TestImgLoader = jt.dataset.DataLoader(
    DA.myImageFloder(all_left_img,train_depth,False),
    batch_size= 1, shuffle = False,num_workers = 1,drop_last = False)

model = basic()

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

modelpath = '/root/siton-gpfs-archive/qiujiaxiong/data_smoke/s2n/checkpoints_real2_fea_rgbd_gradn2_noar/checkpoint_800.tar'
pretrain_dict = jt.load(modelpath)
model.load_state_dict(pretrain_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

out_transform = transforms.Compose([
    transforms.ToPILImage()
])

def test(tgt_img, tgt_depth, tgt_time):
        
        model.eval()
        tgt_img = tgt_img.cuda()
        tgt_time = tgt_time.cuda()
        tgt_depth = tgt_depth.cuda()

        with jt.no_grad():
           output = model(tgt_img,tgt_depth,tgt_time)
           output = output.squeeze()
        #    att = att.squeeze()

        return output,output


def main():
    #------------- TEST ------------------------------------------------------------
    for batch_idx, (tgt_img, tgt_depth, tgt_time, top_pad, right_pad) in enumerate(TestImgLoader):
    
        start_time = time.time()
        result,att = test(tgt_img, tgt_depth, tgt_time)
        print('time = %.2f' %(time.time() - start_time))

        result = result[:,top_pad:,:-right_pad]
        # att = att[top_pad:,:-right_pad]
        # res = res[:,top_pad:,:-right_pad]

        # result = result[:,top_pad:,:-right_pad]
        # result = torchvision.utils.make_grid(result.detach().cpu(), nrow=1, padding=0, normalize=False) 
        result = (result.detach().cpu()*0.5) + 0.5
        # att = (att.detach().cpu()*0.5) + 0.5
        # print(torch.mean((att>0.9).float()))
        # att[att<0.9] = 0.0
        # att[att>0.9] = 1.0
        # result = result*att + 1.0*(1-att)
        
        result = out_transform(result)
        # att = out_transform(att)
        # result = result.filter(ImageFilter.MedianFilter(7))
        result.save(args.savemodel+'/test_img_%06d.jpg'%(batch_idx))
        # att.save(args.savemodel+'/att/test_img_%06d.png'%(batch_idx))
        # res.save(args.savemodel+'/res/test_img_%06d.png'%(batch_idx))
        

if __name__ == '__main__':
   main()






