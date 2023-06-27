_base_ = './default.py'

expname = 'dvgo_ct_gridMLP8_3D_spase_64'
basedir = './logs/'

data = dict(
    datadir='./data/3D/teeth/tooth4/',
    dataset_type='blender',
    white_bkgd=True,
)

