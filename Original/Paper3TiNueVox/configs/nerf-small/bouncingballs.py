_base_ = './default.py'

expname = 'small/dnerf_bouncingballs-400'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='/home/user/Desktop/data/bouncingballs',
    dataset_type='dnerf',
    white_bkgd=True,
)

