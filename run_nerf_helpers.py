import os

import imageio
import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
# from hash_encoding import HashEmbedder
from IPython import embed

# Misc
img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x: (255*np.clip(x,0,1)).astype(np.uint8)

# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, i=1):
    if i == -1:
        return nn.Identity(), 3
    if i == 0:
        embed_kwargs = {
                    'include_input': True,
                    'input_dims': 2,
                    'max_freq_log2': multires-1,
                    'num_freqs': multires,
                    'log_sampling': True,
                    'periodic_fns': [torch.sin, torch.cos],
        }
        embedder_obj = Embedder(**embed_kwargs)
        embed = lambda x, eo=embedder_obj: eo.embed(x)
        return embed, embedder_obj.out_dim
# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=1, skips=[4], use_viewdirs=False):
        """
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)] +
            [nn.Linear(W, output_ch)])
        self.activation = nn.Softplus(beta=100)
        # self.activation = Squareplus()
        print(self.pts_linears)

    def forward(self, x):
        input_pts = x
        outputs = input_pts
        for i, l in enumerate(self.pts_linears):
            if i == len(self.pts_linears)-1:
                outputs = self.pts_linears[i](outputs)
            else:
                outputs = self.pts_linears[i](outputs)
                outputs = self.activation(outputs)
                # outputs = F.relu(outputs)
            if i in self.skips:
                outputs = torch.cat([input_pts, outputs], -1)
        return outputs

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"

        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))

        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))


# # Small NeRF for Hash embeddings
# class NeRFSmall(nn.Module):
#     def __init__(self,
#                  num_layers=3,
#                  hidden_dim=64,
#                  geo_feat_dim=15,
#                  num_layers_color=4,
#                  hidden_dim_color=64,
#                  input_ch=3, input_ch_views=3,
#                  ):
#         super(NeRFSmall, self).__init__()
#
#         self.input_ch = input_ch
#         self.input_ch_views = input_ch_views
#
#         # sigma network
#         self.num_layers = num_layers
#         self.hidden_dim = hidden_dim
#         self.geo_feat_dim = geo_feat_dim
#
#         sigma_net = []
#         for l in range(num_layers):
#             if l == 0:
#                 in_dim = self.input_ch
#             else:
#                 in_dim = hidden_dim
#
#             if l == num_layers - 1:
#                 out_dim = 1  # 1 sigma + 15 SH features for color
#             else:
#                 out_dim = hidden_dim
#
#             sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))
#
#         self.sigma_net = nn.ModuleList(sigma_net)
#         self.softplus = nn.Softplus(beta=100)
#         print(self.sigma_net)
#
#     def forward(self, x):
#         input_pts = x
#
#         # sigma
#         h = input_pts
#         for l in range(self.num_layers):
#             h = self.sigma_net[l](h)
#             if l != self.num_layers - 1:
#                 h = self.softplus(h)
#
#         sigma = h
#         return sigma


class ConeBeam:
    def __init__(self, n, n_detect, s_detect, d_source, angles):
        self.n_detect = n_detect
        self.s_detect = torch.tensor(s_detect)
        self.d_source = d_source
        self.n = torch.tensor(n)
        self.m = torch.tensor((len(angles), n_detect))
        self.angles = angles
        self.geometry_dict = {}

    def _d_detect(self):
        return (
                abs(self.s_detect)
                * self.n_detect
                / self.n[0]
                * torch.sqrt(
                self.d_source * self.d_source
                - (self.n[0] / 2.0) * (self.n[0] / 2.0)
                )
                - self.d_source
            )

    def create_geometry(self):
        # detector positions
        s_range = (torch.arange(self.n_detect).unsqueeze(0) - self.n_detect / 2.0 + 0.5) * self.s_detect

        p_detect_x = s_range
        p_detect_y = -self._d_detect()

        # source position
        p_source_x = 0.0
        p_source_y = self.d_source

        # rotate rays from source to detector over all angles
        pi = torch.acos(torch.zeros(1)).item() * 2.0
        cs = torch.cos(self.angles * pi / 180.0).unsqueeze(1)
        sn = torch.sin(self.angles * pi / 180.0).unsqueeze(1)
        r_p_source_x_ori = p_source_x * cs - p_source_y * sn
        r_p_source_y_ori = p_source_x * sn + p_source_y * cs
        r_p_detect_x_ori = p_detect_x * cs - p_detect_y * sn
        r_p_detect_y_ori = p_detect_x * sn + p_detect_y * cs
        r_dir_x_ori = p_detect_x * cs - p_detect_y * sn - r_p_source_x_ori
        r_dir_y_ori = p_detect_x * sn + p_detect_y * cs - r_p_source_y_ori

        # find intersections of rays with circle for clipping
        max_beta = torch.atan((self.s_detect.abs() * (self.n_detect / 2.0)) / (self.d_source + self._d_detect()))
        radius = self.d_source * torch.sin(max_beta)
        a = r_dir_x_ori * r_dir_x_ori + r_dir_y_ori * r_dir_y_ori
        b = r_p_source_x_ori * r_dir_x_ori + r_p_source_y_ori * r_dir_y_ori
        c = (r_p_source_x_ori * r_p_source_x_ori + r_p_source_y_ori * r_p_source_y_ori - radius * radius)
        ray_length_threshold = 1.0
        discriminant_sqrt = torch.sqrt(
            torch.max(
                b * b - a * c,
                torch.tensor(ray_length_threshold),
            )
        )
        lambda_1 = (-b - discriminant_sqrt) / a
        lambda_2 = (-b + discriminant_sqrt) / a

        # clip ray accordingly
        r_p_source_x = r_p_source_x_ori + lambda_1 * r_dir_x_ori
        r_p_source_y = r_p_source_y_ori + lambda_1 * r_dir_y_ori
        r_dir_x = r_dir_x_ori * (lambda_2 - lambda_1)
        r_dir_y = r_dir_y_ori * (lambda_2 - lambda_1)

        p_source = torch.concat([r_p_source_x[..., None], r_p_source_y[..., None]], dim=-1)
        p_source_ori = torch.concat([r_p_source_x_ori, r_p_source_y_ori], dim=-1)
        p_detect_ori = torch.concat([r_p_detect_x_ori[..., None], r_p_detect_y_ori[..., None]], dim=-1)
        ray_dir = torch.concat([r_dir_x[..., None], r_dir_y[..., None]], dim=-1)
        ray_dir_ori = torch.concat([r_dir_x_ori[..., None], r_dir_y_ori[..., None]], dim=-1)

        geometry_dict = {
            'p_source': p_source,
            'p_source_ori': p_source_ori,
            'p_detect_ori': p_detect_ori,
            'rays_dir': ray_dir,
            'rays_dir_ori': ray_dir_ori,
        }
        self.geometry_dict.update(geometry_dict)

    def preprocessing(self):
        p_detect_ori = self.geometry_dict['p_detect_ori']
        ray_dir_ori = self.geometry_dict['rays_dir_ori']

        Pc = (p_detect_ori[:, int(p_detect_ori.shape[1]/2 -1), :] + p_detect_ori[:, int(p_detect_ori.shape[1]/2), :]) / 2
        Pio = -1.0 * ray_dir_ori
        Pic = torch.tile(Pc[:, None, :], (1, p_detect_ori.shape[1], 1)) - p_detect_ori
        cos_beta = torch.sum(Pio * Pic, dim=-1) / (torch.sqrt(torch.sum(torch.pow(Pio, 2), -1)) * torch.sqrt(torch.sum(torch.pow(Pic, 2), -1)))
        sin_beta = torch.sqrt(1.0 - torch.pow(cos_beta, 2))
        Xi = 0.5 * sin_beta
        Yi_res = 0.5 * cos_beta
        Yi = torch.sqrt(torch.sum(torch.pow(ray_dir_ori, 2), -1)) - Yi_res
        return Xi, Yi

def rotation(points, beta):
    R_M = np.array([
        [np.cos(beta), -np.sin(beta)],
        [np.sin(beta), np.cos(beta)]
    ])
    R_M = torch.FloatTensor(R_M)
    return torch.matmul(points, R_M)

# Ray helpers
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d

def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d

def load_inter_points(basedir):
    print("--------加载交点数据-----------")
    # path1 = os.path.join(basedir, 'all_in_pts.npy')
    # path2 = os.path.join(basedir, 'all_out_pts.npy')
    path1 = os.path.join(basedir, 'all_in_pts_240_512_652.npy')
    path2 = os.path.join(basedir, 'all_out_pts_240_512_652.npy')
    in_points = np.load(path1)
    out_points = np.load(path2)
    return in_points, out_points

def sampling_between_two_points(in_p, out_p, sample_n=256):
    in_p = in_p.numpy()
    out_p = out_p.numpy()
    in_p = np.reshape(in_p, (-1, in_p.shape[-1]))
    out_p = np.reshape(out_p, (-1, out_p.shape[-1]))

    t = np.linspace(0, 1, sample_n)
    t = np.reshape(t, (1, -1))
    t = t * np.ones((in_p.shape[0], sample_n))

    x1 = in_p[:, 0]
    y1 = in_p[:, 1]
    z1 = in_p[:, 2]
    x1 = np.reshape(x1, (-1, 1))
    y1 = np.reshape(y1, (-1, 1))
    z1 = np.reshape(z1, (-1, 1))

    x2 = out_p[:, 0]
    y2 = out_p[:, 1]
    z2 = out_p[:, 2]
    x2 = np.reshape(x2, (-1, 1))
    y2 = np.reshape(y2, (-1, 1))
    z2 = np.reshape(z2, (-1, 1))

    x = (1 - t) * x1 + t * x2
    y = (1 - t) * y1 + t * y2
    z = (1 - t) * z1 + t * z2
    sample_points = np.stack([x, y, z], axis=-1)
    sample_points = torch.FloatTensor(sample_points)

    # start = sample_points[:, 0, :]
    # end = sample_points[:, -1, :]
    # dist = np.sqrt(np.sum(np.power(end - start, 2), -1))
    # dist = np.reshape(dist, (-1, 1))
    # return sample_points, dist
    return sample_points

def get_testing_rays():
    eps_time = time.time()
    all_in_pts = []
    all_out_pts = []

    Y = 965
    y_stepsize = Y / 965
    y = np.arange(0, Y, y_stepsize)
    y = (y - Y * 0.5) + y_stepsize / 2

    Z = 652 * 2
    z_stepsize = Z / 1304.0
    z = np.arange(0, Z, z_stepsize)
    z = (z - Z * 0.5) + z_stepsize / 2

    for i in range(len(y)):
        y_th = y[i]
        in_pts = np.stack([-np.ones_like(z) * 652, np.ones_like(z) * y_th, z], -1)
        out_pts = np.stack([np.ones_like(z) * 652, np.ones_like(z) * y_th, z], -1)
        all_in_pts.append(in_pts)
        all_out_pts.append(out_pts)
    all_in_pts = np.array(all_in_pts)
    all_out_pts = np.array(all_out_pts)
    all_in_pts = torch.FloatTensor(all_in_pts) / 1304.0
    all_out_pts = torch.FloatTensor(all_out_pts) / 1304.0
    eps_time = time.time() - eps_time
    print('get_training_rays: finish (eps time:', eps_time, 'sec)')
    return all_in_pts, all_out_pts

# 2D
def get_training_rays(sample=0, device=0, sparse=48):
    sample_points = np.load('./data/2D/sample_points_128.npy')
    sinogram = np.load('./data/2D/samples/sample_{}_sino.npy'.format(sample))

    # z = np.zeros((128, 1024, 513, 1))
    # sample_points = np.concatenate([sample_points, z], -1)
    sample_points = torch.FloatTensor(sample_points)
    sinogram = torch.FloatTensor(sinogram)

    index = np.round(np.linspace(start = 0, stop = sinogram.shape[0], num = sparse+1)[:-1])
    # index = np.arange(0, sinogram.shape[0], sparse)

    sinogram = sinogram[index]
    sample_points = sample_points[index]

    sample_points = sample_points.to(device)
    sinogram = sinogram.to(device)
    print('load sample_{}'.format(sample))
    print('sinogram shape is: {}, sample_points shape is: {}'.format(sinogram.shape, sample_points.shape))
    return sample_points, sinogram

# 稀疏到32个视角
def get_training_rays2D(sample=0):
    # sinogram = np.load(gzip.GzipFile(os.path.join(basedir, "Sinogram_batch1.npy.gz"), "r", ))
    sample_points = np.load('./data/2D/sample_points_128.npy')
    sinogram = np.load('./data/2D/samples/sample_{}_sino.npy'.format(sample))

    index = np.arange(0, sinogram.shape[0], 2)
    sinogram = sinogram[index]
    sample_points = sample_points[index]
    sinogram = torch.FloatTensor(sinogram)
    sample_points = torch.FloatTensor(sample_points)
    print('load sample_{}  and   0_sinogram_gt'.format(sample))
    print('sinogram shape is: {}, sample_points shape is: {}'.format(sinogram.shape, sample_points.shape))
    return sample_points, sinogram

# 稀疏到16个视角
def get_training_rays2D_1():
    # sinogram = np.load(gzip.GzipFile(os.path.join(basedir, "Sinogram_batch1.npy.gz"), "r", ))
    sample_points_name = 'sample_points_128'
    sample_points = np.load('./data/2D/{}.npy'.format(sample_points_name))
    sinogram = np.load('./data/2D/0_sinogram_gt.npy')

    index = np.arange(0, sinogram.shape[0], 8)
    sinogram = sinogram[index]
    sample_points = sample_points[index]
    sinogram = torch.FloatTensor(sinogram)
    sample_points = torch.FloatTensor(sample_points)
    print('load {}  and   0_sinogram_gt'.format(sample_points_name))
    print('sinogram shape is: {}, sample_points shape is: {}'.format(sinogram.shape, sample_points.shape))
    return sample_points, sinogram

# 稀疏到8个视角
def get_training_rays2D_2():
    # sinogram = np.load(gzip.GzipFile(os.path.join(basedir, "Sinogram_batch1.npy.gz"), "r", ))
    sample_points_name = 'sample_points_128'
    sample_points = np.load('./data/2D/{}.npy'.format(sample_points_name))
    sinogram = np.load('./data/2D/0_sinogram_gt.npy')

    index = np.arange(0, sinogram.shape[0], 16)
    sinogram = sinogram[index]
    sample_points = sample_points[index]
    sinogram = torch.FloatTensor(sinogram)
    sample_points = torch.FloatTensor(sample_points)
    print('load {}  and   0_sinogram_gt'.format(sample_points_name))
    print('sinogram shape is: {}, sample_points shape is: {}'.format(sinogram.shape, sample_points.shape))
    return sample_points, sinogram

def get_training_rays2D_3():
    # angle = torch.arange(-0.5 * torch.pi, 1.5 * torch.pi, (2 * torch.pi)/32.0)
    angle = np.linspace(0, 2 * np.pi, 1025)[:-1]
    angle = torch.FloatTensor(angle)

    sample_points_name = 'sample_points_128'
    sample_points = np.load('./data/2D/{}.npy'.format(sample_points_name))
    sinogram = np.load('./data/2D/0_sinogram_gt.npy')

    index = np.arange(0, sample_points.shape[0], 4)
    sample_points = sample_points[index]
    sinogram = sinogram[index]
    sample_points = torch.FloatTensor(sample_points)
    start_points = sample_points[0]
    start_points = torch.FloatTensor(start_points)

    new_points = []
    for i in range(len(angle)):
        aa = rotation(start_points, angle[i])
        new_points.append(aa[None, ...])
    new_points = torch.concat(new_points, dim=0)
    return new_points, sinogram

def get_training_rays2D_4():
    # angle = torch.arange(-0.5 * torch.pi, 1.5 * torch.pi, (2 * torch.pi)/32.0)
    angle = np.linspace(0, 2 * np.pi, 1025)[:-1]
    angle = torch.FloatTensor(angle)

    sample_points_name = 'sample_points_128'
    sample_points = np.load('./data/2D/{}.npy'.format(sample_points_name))
    sinogram = np.load('./data/2D/0_sinogram_gt.npy')

    index = np.arange(0, sample_points.shape[0], 8)
    sample_points = sample_points[index]
    sinogram = sinogram[index]
    sample_points = torch.FloatTensor(sample_points)
    start_points = sample_points[0]
    start_points = torch.FloatTensor(start_points)

    new_points = []
    for i in range(len(angle)):
        aa = rotation(start_points, angle[i])
        new_points.append(aa[None, ...])
    new_points = torch.concat(new_points, dim=0)
    return new_points, sinogram

def get_training_rays2D_5():
    # sinogram = np.load(gzip.GzipFile(os.path.join(basedir, "Sinogram_batch1.npy.gz"), "r", ))
    sample_points_name = 'sample_points_128'
    sample_points = np.load('./data/2D/{}.npy'.format(sample_points_name))
    sinogram = np.load('./data/2D/0_sinogram_gt.npy')

    index = np.arange(0, sinogram.shape[0], 8)
    sinogram = sinogram[index]
    sample_points = sample_points[index]
    sinogram = torch.FloatTensor(sinogram)
    sample_points = torch.FloatTensor(sample_points)
    print('load {}  and   0_sinogram_gt'.format(sample_points_name))
    print('sinogram shape is: {}, sample_points shape is: {}'.format(sinogram.shape, sample_points.shape))
    return sample_points, sinogram

# 3D
def load_sparse_project_data(basedir, factor1=1, train=True):
    print("--------加载投影数据-----------")
    # basedir = os.path.join(basedir, 'project_data')
    basedir = os.path.join(basedir, 'project_data_half')

    data_path = [os.path.join(basedir, p) for p in sorted(os.listdir(basedir)) if p.endswith('npy')]
    data_path = np.array(data_path)
    if factor1 > 1 and train:
        index1 = np.arange(0, len(data_path))
        index2 = np.arange(0, len(data_path), factor1)
        index = [idx for idx in index1 if idx not in index2]
        data_path = data_path[index]
    elif factor1 > 1 and not train:
        index = np.arange(0, len(data_path), factor1)
        data_path = data_path[index]
    project_data = [np.load(f) for f in data_path]
    project_data = np.array(project_data)
    # from matplotlib import pyplot as plt
    # fig = plt.figure()
    # for i in tqdm(range(len(project_data)), total=len(project_data), smoothing=0.9):
    #     data = project_data[i]
    #     flag = (data > 2300).astype(np.int64)
    #     data = data * flag
    #     plt.imshow(data, cmap='gray')
    #     fig.savefig('../logs/nerf_synthetic/dvgo_lego/gt_projection1/{:3d}.png'.format(i), bbox_inches="tight")
    project_data = project_data / project_data.max()
    return project_data


def get_training_rays3D(basedir, train=True):
    eps_time = time.time()
    factor1 = 5
    factor2 = 8   # 将192稀疏到168  减少24个视角
    # factor2 = 4  # 将192稀疏到144  减少48个视角
    # factor2 = 2  # 将192稀疏到96  减少96个视角
    project_data = load_sparse_project_data(basedir, factor1=factor1, train=train)
    in_points, out_points = load_inter_points(basedir)

    if factor1 > 1 and train:
        index1 = np.arange(0, len(in_points))
        index2 = np.arange(0, len(in_points), factor1)
        index = [idx for idx in index1 if idx not in index2]
        in_points = in_points[index]
        out_points = out_points[index]
    elif factor1 > 1 and not train:
        index = np.arange(0, len(in_points), factor1)
        in_points = in_points[index]
        out_points = out_points[index]

    if factor2 > 1 and train:
        train_total_num = len(in_points)
        index1 = np.arange(0, train_total_num)
        index2 = np.arange(0, train_total_num, factor2)
        index = [idx for idx in index1 if idx not in index2]
        in_points = in_points[index]
        out_points = out_points[index]
        project_data = project_data[index]


    in_points = in_points / 200.0
    out_points = out_points / 200.0
    project_data = torch.FloatTensor(project_data)
    in_points = torch.FloatTensor(in_points)
    out_points = torch.FloatTensor(out_points)

    eps_time = time.time() - eps_time
    print('get_training_rays: finish (eps time:', eps_time, 'sec)')
    print(project_data.shape, in_points.shape, out_points.shape)
    return project_data, in_points, out_points

# 加载128_512_512投影和交点数据, 稀疏到32
def get_train_data_3d(basedir, train=True):
    eps_time = time.time()
    print("==============加载投影数据==============")
    projdir = os.path.join(basedir, 'projection_512_512_128')
    data_path = [os.path.join(projdir, p) for p in sorted(os.listdir(projdir)) if p.endswith('npy')]
    data_path = np.array(data_path)
    project_data = [np.load(f) for f in data_path]
    project_data = np.array(project_data)
    project_data = project_data / project_data.max()
    print("==============加载交点数据==============")
    path1 = os.path.join(basedir, 'all_in_pts_512_512_128.npy')
    path2 = os.path.join(basedir, 'all_out_pts_512_512_128.npy')
    in_points = np.load(path1) / 200.0
    out_points = np.load(path2) / 200.0

    index1 = np.arange(0, len(in_points), 4)
    in_points = in_points[index1]
    out_points = out_points[index1]
    project_data = project_data[index1]

    project_data = torch.FloatTensor(project_data)
    in_points = torch.FloatTensor(in_points)
    out_points = torch.FloatTensor(out_points)

    eps_time = time.time() - eps_time
    print(f'project_data: {project_data.shape}, in_points: {in_points.shape}, out_points: {out_points.shape}')
    print('get_training_data: finish (eps time:', eps_time, 'sec)')
    return project_data, in_points, out_points

# 加载128_512_512投影和交点数据, 稀疏到16
def get_train_data_3d_1(basedir, train=True):
    eps_time = time.time()
    print("==============加载投影数据==============")
    projdir = os.path.join(basedir, 'projection_512_512_128')
    data_path = [os.path.join(projdir, p) for p in sorted(os.listdir(projdir)) if p.endswith('npy')]
    data_path = np.array(data_path)
    project_data = [np.load(f) for f in data_path]
    project_data = np.array(project_data)
    project_data = project_data / project_data.max()
    print("==============加载交点数据==============")
    path1 = os.path.join(basedir, 'all_in_pts_512_512_128.npy')
    path2 = os.path.join(basedir, 'all_out_pts_512_512_128.npy')
    in_points = np.load(path1) / 200.0
    out_points = np.load(path2) / 200.0

    index1 = np.arange(0, len(in_points), 8)
    in_points = in_points[index1]
    out_points = out_points[index1]
    project_data = project_data[index1]

    project_data = torch.FloatTensor(project_data)
    in_points = torch.FloatTensor(in_points)
    out_points = torch.FloatTensor(out_points)

    eps_time = time.time() - eps_time
    print(f'project_data: {project_data.shape}, in_points: {in_points.shape}, out_points: {out_points.shape}')
    print('get_training_data: finish (eps time:', eps_time, 'sec)')
    return project_data, in_points, out_points

# 加载128_512_512投影和交点数据, 稀疏到32
def get_train_data_3d_2(basedir, train=True):
    eps_time = time.time()
    print("==============加载投影数据==============")
    projdir = os.path.join(basedir, 'projection_512_512_128')
    data_path = [os.path.join(projdir, p) for p in sorted(os.listdir(projdir)) if p.endswith('npy')]
    data_path = np.array(data_path)
    project_data = [np.load(f) for f in data_path]
    project_data = np.array(project_data)
    project_data = project_data / project_data.max()
    print("==============加载交点数据==============")
    path1 = os.path.join(basedir, 'all_in_pts_512_512_128.npy')
    path2 = os.path.join(basedir, 'all_out_pts_512_512_128.npy')
    in_points = np.load(path1)
    out_points = np.load(path2)

    index1 = np.arange(0, len(in_points), 4)
    in_points = in_points[index1]
    out_points = out_points[index1]
    project_data = project_data[index1]

    out_points_x = out_points[..., 0]
    out_points_y = out_points[..., 1]
    out_points_z = out_points[..., 2]
    out_pts_x = (out_points_x - out_points_x.min()) / (out_points_x.max() - out_points_x.min()) * 652
    out_pts_y = (out_points_y - out_points_y.min()) / (out_points_y.max() - out_points_y.min()) * 965
    out_pts_z = (out_points_z - out_points_z.min()) / (out_points_z.max() - out_points_z.min()) * 652
    out_pts = np.concatenate([out_pts_x[..., None], out_pts_y[..., None], out_pts_z[..., None]], axis=-1)

    in_points_x = in_points[..., 0]
    in_points_y = in_points[..., 1]
    in_points_z = in_points[..., 2]
    in_pts_x = (in_points_x - out_points_x.min()) / (out_points_x.max() - out_points_x.min()) * 652
    in_pts_y = (in_points_y - out_points_y.min()) / (out_points_y.max() - out_points_y.min()) * 965
    in_pts_z = (in_points_z - out_points_z.min()) / (out_points_z.max() - out_points_z.min()) * 652
    in_pts = np.concatenate([in_pts_x[..., None], in_pts_y[..., None], in_pts_z[..., None]], axis=-1)

    project_data = torch.FloatTensor(project_data)
    in_pts = torch.FloatTensor(in_pts)
    out_pts = torch.FloatTensor(out_pts)

    # in_points_x = in_points[..., 0:1] + 767
    # in_points_y = in_points[..., 1:2] + 546
    # in_points_z = in_points[..., 2:3] + 767
    # in_points = torch.concat([in_points_x, in_points_y, in_points_z], dim=-1)
    # out_points_x = out_points[..., 0:1] + 767
    # out_points_y = out_points[..., 1:2] + 546
    # out_points_z = out_points[..., 2:3] + 767
    # out_points = torch.concat([out_points_x, out_points_y, out_points_z], dim=-1)

    eps_time = time.time() - eps_time
    print(f'project_data: {project_data.shape}, in_points: {in_points.shape}, out_points: {out_points.shape}')
    print('get_training_data: finish (eps time:', eps_time, 'sec)')
    return project_data, in_pts, out_pts

# 加载128_512_512投影和交点数据, 稀疏到16
def get_train_data_3d_3(basedir, train=True):
    eps_time = time.time()
    print("==============加载投影数据==============")
    projdir = os.path.join(basedir, 'projection_512_512_128')
    data_path = [os.path.join(projdir, p) for p in sorted(os.listdir(projdir)) if p.endswith('npy')]
    data_path = np.array(data_path)
    project_data = [np.load(f) for f in data_path]
    project_data = np.array(project_data)
    project_data = project_data / project_data.max()
    print("==============加载交点数据==============")
    path1 = os.path.join(basedir, 'all_in_pts_512_512_128.npy')
    path2 = os.path.join(basedir, 'all_out_pts_512_512_128.npy')
    in_points = np.load(path1)
    out_points = np.load(path2)

    index1 = np.arange(0, len(in_points), 8)
    in_points = in_points[index1]
    out_points = out_points[index1]
    project_data = project_data[index1]

    out_points_x = out_points[..., 0]
    out_points_y = out_points[..., 1]
    out_points_z = out_points[..., 2]
    out_pts_x = (out_points_x - out_points_x.min()) / (out_points_x.max() - out_points_x.min()) * 652
    out_pts_y = (out_points_y - out_points_y.min()) / (out_points_y.max() - out_points_y.min()) * 965
    out_pts_z = (out_points_z - out_points_z.min()) / (out_points_z.max() - out_points_z.min()) * 652
    out_pts = np.concatenate([out_pts_x[..., None], out_pts_y[..., None], out_pts_z[..., None]], axis=-1)

    in_points_x = in_points[..., 0]
    in_points_y = in_points[..., 1]
    in_points_z = in_points[..., 2]
    in_pts_x = (in_points_x - out_points_x.min()) / (out_points_x.max() - out_points_x.min()) * 652
    in_pts_y = (in_points_y - out_points_y.min()) / (out_points_y.max() - out_points_y.min()) * 965
    in_pts_z = (in_points_z - out_points_z.min()) / (out_points_z.max() - out_points_z.min()) * 652
    in_pts = np.concatenate([in_pts_x[..., None], in_pts_y[..., None], in_pts_z[..., None]], axis=-1)

    project_data = torch.FloatTensor(project_data)
    in_pts = torch.FloatTensor(in_pts)
    out_pts = torch.FloatTensor(out_pts)

    # in_points_x = in_points[..., 0:1] + 767
    # in_points_y = in_points[..., 1:2] + 546
    # in_points_z = in_points[..., 2:3] + 767
    # in_points = torch.concat([in_points_x, in_points_y, in_points_z], dim=-1)
    # out_points_x = out_points[..., 0:1] + 767
    # out_points_y = out_points[..., 1:2] + 546
    # out_points_z = out_points[..., 2:3] + 767
    # out_points = torch.concat([out_points_x, out_points_y, out_points_z], dim=-1)

    eps_time = time.time() - eps_time
    print(f'project_data: {project_data.shape}, in_points: {in_points.shape}, out_points: {out_points.shape}')
    print('get_training_data: finish (eps time:', eps_time, 'sec)')
    return project_data, in_pts, out_pts

def get_train_data_3d_4(basedir, train=True):
    eps_time = time.time()
    print("==============加载投影数据==============")
    projdir = os.path.join(basedir, 'projection_512_512_128_1')
    data_path = [os.path.join(projdir, p) for p in sorted(os.listdir(projdir)) if p.endswith('npy')]
    data_path = np.array(data_path)
    project_data = [np.load(f) for f in data_path]
    project_data = np.array(project_data)
    project_data = project_data / project_data.max()
    print("==============加载交点数据==============")
    path1 = os.path.join(basedir, 'all_in_pts_512_512_128_1.npy')
    path2 = os.path.join(basedir, 'all_out_pts_512_512_128_1.npy')
    in_points = np.load(path1)
    out_points = np.load(path2)

    index1 = np.arange(0, len(in_points), 4)
    in_points = in_points[index1]
    out_points = out_points[index1]
    project_data = project_data[index1]

    out_points_x = out_points[..., 0]
    out_points_y = out_points[..., 1]
    out_points_z = out_points[..., 2]
    out_pts_x = out_points_x - out_points_x.min()
    out_pts_y = out_points_y - out_points_y.min()
    out_pts_z = out_points_z - out_points_z.min()
    out_pts = np.concatenate([out_pts_x[..., None], out_pts_y[..., None], out_pts_z[..., None]], axis=-1)

    in_points_x = in_points[..., 0]
    in_points_y = in_points[..., 1]
    in_points_z = in_points[..., 2]
    in_pts_x = in_points_x - out_points_x.min()
    in_pts_y = in_points_y - out_points_y.min()
    in_pts_z = in_points_z - out_points_z.min()
    in_pts = np.concatenate([in_pts_x[..., None], in_pts_y[..., None], in_pts_z[..., None]], axis=-1)

    # np.savetxt('aaa.txt', np.reshape(in_pts[0], (-1, 3)))
    # np.savetxt('bbb.txt', np.reshape(out_pts[0], (-1, 3)))

    project_data = torch.FloatTensor(project_data)
    in_pts = torch.FloatTensor(in_pts)
    out_pts = torch.FloatTensor(out_pts)

    # in_points_x = in_points[..., 0:1] + 767
    # in_points_y = in_points[..., 1:2] + 546
    # in_points_z = in_points[..., 2:3] + 767
    # in_points = torch.concat([in_points_x, in_points_y, in_points_z], dim=-1)
    # out_points_x = out_points[..., 0:1] + 767
    # out_points_y = out_points[..., 1:2] + 546
    # out_points_z = out_points[..., 2:3] + 767
    # out_points = torch.concat([out_points_x, out_points_y, out_points_z], dim=-1)

    eps_time = time.time() - eps_time
    print(f'project_data: {project_data.shape}, in_points: {in_pts.shape}, out_points: {out_pts.shape}')
    print('get_training_data: finish (eps time:', eps_time, 'sec)')
    return project_data, in_pts, out_pts

# 加载128_512_512投影和交点数据, 128视角
def get_train_data_3d_5(basedir, train=True):
    eps_time = time.time()
    print("==============加载投影数据==============")
    projdir = os.path.join(basedir, 'data_tooth')
    data_path = [os.path.join(projdir, p) for p in sorted(os.listdir(projdir)) if p.endswith('.tif')]
    data_path = np.array(data_path)
    project_data = [imageio.imread(f) for f in data_path]
    project_data = np.array(project_data)
    project_data = project_data / project_data.max()
    print("==============加载交点数据==============")
    path1 = os.path.join(basedir, 'all_in_pts_512_512_128.npy')
    path2 = os.path.join(basedir, 'all_out_pts_512_512_128.npy')
    in_points = np.load(path1) / 200.0
    out_points = np.load(path2) / 200.0

    project_data = torch.FloatTensor(project_data)
    in_points = torch.FloatTensor(in_points)
    out_points = torch.FloatTensor(out_points)

    eps_time = time.time() - eps_time
    print(f'project_data: {project_data.shape}, in_points: {in_points.shape}, out_points: {out_points.shape}')
    print('get_training_data: finish (eps time:', eps_time, 'sec)')
    return project_data, in_points, out_points

def get_testing_rays3D():
    eps_time = time.time()
    all_in_pts = []
    all_out_pts = []

    # Y = 872  # tooth1
    # Y = 918  # tooth2
    # Y = 862  # tooth3
    Y = 965  # tooth4
    y_stepsize = 1
    y = np.arange(0, Y, y_stepsize)
    y = (y - Y * 0.5) + y_stepsize / 2

    Z = 652 * 2
    z_stepsize = Z / 1304.0
    z = np.arange(0, Z, z_stepsize)
    z = (z - Z * 0.5) + z_stepsize / 2

    for i in range(len(y)):
        y_th = y[i]
        in_pts = np.stack([-np.ones_like(z) * 652, np.ones_like(z) * y_th, z], -1)
        out_pts = np.stack([np.ones_like(z) * 652, np.ones_like(z) * y_th, z], -1)
        all_in_pts.append(in_pts)
        all_out_pts.append(out_pts)
    all_in_pts = np.array(all_in_pts)
    all_out_pts = np.array(all_out_pts)
    test_in_pts = torch.FloatTensor(all_in_pts) / 200.0
    test_out_pts = torch.FloatTensor(all_out_pts) / 200.0
    eps_time = time.time() - eps_time
    print('get_testing_rays: finish (eps time:', eps_time, 'sec)')
    return test_in_pts, test_out_pts

if __name__ == '__main__':
    # n = [512, 512]
    # n_detect = 1024
    # s_detect = -1
    # d_source = 1422.1301
    # angles = torch.linspace(-90, 270, 129)[:-1]
    # device = 'cpu'
    # conebeam = ConeBeam(n, n_detect, s_detect, d_source, angles, device)
    # conebeam.create_geometry()
    # conebeam.preprocessing()
    get_testing_rays3D()



