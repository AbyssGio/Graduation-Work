_base_ = './default2.py'

expname = 'ours/scarlar_vel_fean'
basedir = './logs/scarlar'

data = dict(
    datadir='/home/ubuntu/Downloads/pinf_data/data1',
    dataset_type='dnerf',
    white_bkgd=False,
)