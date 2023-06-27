from copy import deepcopy

expname = None                    # experiment name
basedir = './logs/'               # where to store ckpts and logs

''' Template of data options
'''
data = dict(
    datadir=None,                 # path to dataset root folder
    dataset_type=None,            # blender | nsvf | blendedmvs | tankstemple | deepvoxels | co3d
)

''' Template of training options
'''
train_2D = dict(
    N_iters=1000,                # number of optimization steps
    N_rand=8192,                  # batch size (number of random rays per optimization step)
    lrate_density=1e-1,           # lr of density voxel grid
    lrate_mlps=1e-3,               # lr of the mlp to preduct view-dependent color
    lrate_decay=20,               # lr decay by 0.1 after every lrate_decay*1000 steps
    decay_base=2000,
    pg_scale=[],                  # checkpoints for progressive scaling
    skip_zero_grad_fields=['density'],     # the variable name to skip optimizing parameters w/ zero grad in each iteration
)

train_3D = dict(
    N_iters=1,                    # number of optimization steps
    N_rand=8192,                  # batch size (number of random rays per optimization step)
    lrate_density=1e-1,           # lr of density voxel grid
    lrate_mlps=1e-3,               # lr of the mlp to preduct view-dependent color
    lrate_decay=20,               # lr decay by 0.1 after every lrate_decay*1000 steps
    decay_base=5000,
    pg_scale=[],                  # checkpoints for progressive scaling
    skip_zero_grad_fields=['density'],
)

''' Template of model and rendering options
'''
model_and_render_2D = dict(
    num_voxels=300**2,           # expected number of voxel
    num_voxels_base=300**2,      # to rescale delta distance
    density_type='DenseGrid',     # DenseGrid, TensoRFGrid
    density_config=dict(),
    mlp_depth=8,                  # depth of the colors MLP (there are rgbnet_depth-1 intermediate features)
    mlp_width=256,                # width of the colors MLP
    alpha_init=1e-6,              # set the alpha values everywhere at the begin of training
    multires=10,
    i_embed=0,
    skips=[4],
)

model_and_render_3D = dict(
    num_voxels=300**3,           # expected number of voxel
    num_voxels_base=300**3,      # to rescale delta distance
    density_type='DenseGrid',     # DenseGrid, TensoRFGrid
    density_config=dict(),
    mlp_depth=6,                  # depth of the colors MLP (there are rgbnet_depth-1 intermediate features)
    mlp_width=256,                # width of the colors MLP
    alpha_init=1e-6,              # set the alpha values everywhere at the begin of training
    multires=10,
    i_embed=0,
    skips=[3],
)

del deepcopy
