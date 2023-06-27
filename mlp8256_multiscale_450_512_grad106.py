import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import torch

import sys
import numpy as np
import imageio
import random
import time
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
from IPython import embed
import matplotlib.pyplot as plt

np.random.seed(0)
DEBUG = False

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

def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3

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
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])

        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.rgb_linear = nn.Linear(W // 2, 1)
        else:
            self.output_linear = nn.Linear(W, output_ch)
        self.softplus = nn.Softplus(beta=100)
        print(self.pts_linears)
        print(self.views_linears)

    def forward(self, x):
        input_pts, input_erea = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = self.softplus(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_erea], -1)
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = self.softplus(h)
            outputs = self.rgb_linear(h)
        else:
            outputs = self.output_linear(h)

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

def l2_error(X, X_ref, relative=False, squared=False, use_magnitude=True):
    X_flat = np.reshape(X, (1, -1))
    X_ref_flat = np.reshape(X_ref, (1, -1))

    if squared:
        err = (np.linalg.norm(x=(X_flat - X_ref_flat), ord=2) ** 2)
    else:
        err = np.linalg.norm(x=(X_flat - X_ref_flat), ord=2)

    if relative:
        if squared:
            err = err / (np.linalg.norm(x=X_ref_flat, ord=2) ** 2)
        else:
            err = err / np.linalg.norm(x=X_ref_flat, ord=2)
    return err

def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret

def run_network(inputs, area, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if area is not None:
        input_eara = area
        input_dirs_flat = torch.reshape(area, [-1, input_eara.shape[-1]])
        embedded_erea = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_erea], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs

def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 1
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    grad_vars = list(model.parameters())

    network_query_fn = lambda inputs, erea, network_fn: run_network(inputs, erea, network_fn,
                                                              embed_fn=embed_fn,
                                                              embeddirs_fn=embeddirs_fn,
                                                              netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    sample = args.sample
    basedir = args.basedir
    expname = args.expname
    root_dir = os.path.join(basedir, 'lung_sample_{}'.format(sample))
    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(root_dir, expname, f) for f in sorted(os.listdir(os.path.join(root_dir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'network_fn': model,
    }

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer

def render_rays(rays, network_fn, network_query_fn, scale_factor=0.0352):
    """
        rays: [N, 513, 2]
        grad: [N, 513, 1]
    """
    start_points = rays[:, 0, :]
    end_points = rays[:, -1, :]
    rays_o = start_points
    rays_d = end_points - start_points

    N_samples = random.randint(450+1, 512+1)
    near = torch.zeros((rays.shape[0], 1))
    far = torch.ones((rays.shape[0], 1))
    t_vals = torch.linspace(0., 1., steps=N_samples)
    z_vals = near * (1. - t_vals) + far * t_vals
    z_vals = z_vals.expand([rays.shape[0], N_samples])

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 2]

    mid_pts = (pts[:, 1:, :] + pts[:, :-1, :]) / 2.0                                  # [N_rays, N_samples-1, 2]
    dirs = pts[:, 1:, :] - pts[:, :-1, :]                                             # [N_rays, N_samples-1, 2]
    dists = torch.sqrt(torch.sum(torch.pow(dirs, 2), dim=-1))* 255.5                  # [N_rays, N_samples-1, 1]

    ox = torch.tensor([[1.0], [0.0]])
    oy = torch.tensor([[0.0], [1.0]])

    ww = torch.abs(torch.matmul(dirs, ox))   # [N_rays, N_samples-1, 1]
    hh = torch.abs(torch.matmul(dirs, oy))   # [N_rays, N_samples-1, 1]
    erea = torch.concat([ww, hh], dim=-1)

    mid_pts = mid_pts.requires_grad_(True)

    raw = network_query_fn(mid_pts, erea, network_fn)

    d_output = torch.ones_like(raw, requires_grad=False, device=raw.device)
    gradients = torch.autograd.grad(
        outputs=raw,
        inputs=mid_pts,
        grad_outputs=d_output,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    [w, h] = gradients.shape[:-1]
    grad_x = torch.sum(torch.abs(gradients[:, :, 0]))
    grad_y = torch.sum(torch.abs(gradients[:, :, 1]))
    l1_loss = ((grad_x + grad_y) / (w * h)) * 0.000001

    raw2pixel = torch.reshape(raw, (raw.shape[0], raw.shape[1]))

    one_detector_pixel = torch.sum(raw2pixel * dists, dim=-1, keepdim=True) * scale_factor
    return one_detector_pixel, raw2pixel, l1_loss

def test_render_rays(rays, network_fn, network_query_fn):
    w = rays[-1, 0, 0] - rays[-2, 0, 0]
    h = rays[0, 0, 1] - rays[0, 1, 1]
    w = torch.tile(w[None, None, None], (rays.shape[0], rays.shape[1], 1))
    h = torch.tile(h[None, None, None], (rays.shape[0], rays.shape[1], 1))
    erea = torch.concat([w, h], dim=-1)
    raw = network_query_fn(rays, erea, network_fn)
    raw2pixel = torch.reshape(raw, (raw.shape[0], raw.shape[1]))
    return raw2pixel

def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument("--expname", type=str, default='MLP8256_Multiscale_470512_grad106_Angle16', help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',  help='where to store ckpts and logs')

    # training options
    parser.add_argument("--sample", type=int, default=0, help='layers in network')
    parser.add_argument("--sparse", type=int, default=128, help='layers in network')
    parser.add_argument("--gpu", type=int, default=0, help='layers in network')
    parser.add_argument("--netdepth", type=int, default=8, help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, help='channels per layer')
    parser.add_argument("--lrate", type=float, default=5e-4, help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*1024,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*1024,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    # parser.add_argument("--use_viewdirs", action='store_true', help='use full 5D input instead of 3D')
    parser.add_argument("--use_viewdirs", type=bool, default=True, help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000, 
                        help='frequency of render_poses video saving')

    return parser

def train(args):
    # Load data
    sample = args.sample
    sample_points_name = 'sample_points_128'
    sample_points = np.load('./data/2D/{}.npy'.format(sample_points_name))
    sinogram = np.load('./data/2D/lung_samples/sample_{}_sino.npy'.format(sample))

    index = np.round(np.linspace(start=0, stop=sinogram.shape[0], num=args.sparse + 1)[:-1]).astype(int)
    sinogram = sinogram[index]
    sample_points = sample_points[index]
    sinogram = torch.FloatTensor(sinogram)
    sample_points = torch.FloatTensor(sample_points)
    print('load {}  and   0_sinogram_gt'.format(sample_points_name))
    print('sinogram shape is: {}, sample_points shape is: {}'.format(sinogram.shape, sample_points.shape))

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    root_dir = os.path.join(basedir, 'lung_sample_{}'.format(sample))
    os.makedirs(root_dir, exist_ok=True)
    result_dir = os.path.join(root_dir, expname)
    os.makedirs(result_dir, exist_ok=True)
    rec_dir = os.path.join(result_dir, 'reconstruction')
    os.makedirs(os.path.join(result_dir, 'reconstruction'), exist_ok=True)

    os.system('cp %s %s' % ('./mlp8256_multiscale_450_512_grad106.py', result_dir))

    def log_string(out_str):
        logging.write(out_str + '\n')
        logging.flush()
    logging = open(os.path.join(result_dir, 'log_train.txt'), 'a')

    print('=============创建网格网络===================')
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)

    H = 512
    W = 512
    u, v = torch.meshgrid(torch.linspace(0, H, H), torch.linspace(0, W, W))
    pts = torch.stack([(u - W * 0.5), -(v - H * 0.5)], -1) / 256.0
    y = pts[:, :, 0]
    x = pts[:, :, 1]
    r = torch.sqrt(torch.pow(x, 2) + torch.pow(y, 2))
    flag = ((r - 1.0) <= 0.0)
    flag = (flag.cpu().numpy()).astype(np.int64)

    phantom = np.load('./data/2D/lung_samples/sample_{}_phantom.npy'.format(sample))
    vmax = phantom.max()
    vmin = phantom.min()

    torch.cuda.empty_cache()
    print('=============Training Begining=============')
    batch_idx = np.arange(0, sinogram.shape[0])
    N_iters = 1000
    best_rmse = 100.0
    time0 = time.time()
    for global_step in trange(start, N_iters):
        np.random.shuffle(batch_idx)
        total_loss = 0.0
        total_loss1 = 0.0
        step_size = 512
        batch_size = sample_points.shape[1]
        for j in range(len(batch_idx)):
            idx = batch_idx[j]
            sin_row = sinogram[idx]
            batch_target = torch.reshape(sin_row, (-1, 1))

            batch_pts = sample_points[idx]
            for k in range((int(np.ceil(batch_size / step_size)))):
                points = batch_pts[k * step_size:(k + 1) * step_size].to(device)
                target = batch_target[k * step_size:(k + 1) * step_size].to(device)
                one_detector_pixel, _, l1_loss = render_rays(points, **render_kwargs_train)

                optimizer.zero_grad()
                loss = img2mse(one_detector_pixel, target) + l1_loss
                loss.backward()
                optimizer.step()

                total_loss += loss.detach()
                total_loss1 += l1_loss.detach()
                # NOTE: IMPORTANT!
                ###   update learning rate   ###
                decay_rate = 0.1
                decay_steps = args.lrate_decay * 1000
                new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lrate
                ################################

        tatol_loss = total_loss / (sample_points.shape[0] * int(np.ceil(batch_size / step_size)))
        tatol_loss1 = total_loss1 / (sample_points.shape[0] * int(np.ceil(batch_size / step_size)))
        eps_time = time.time() - time0
        eps_time_str = f'{eps_time // 3600:02.0f}:{eps_time // 60 % 60:02.0f}:{eps_time % 60:02.0f}'
        tqdm.write("\tEpoch {} Avarage loss is:{}, l1_loss is {}, take time: {} seconds".format(global_step + 1, tatol_loss, tatol_loss1, eps_time_str))
        log_string("\tEpoch {} Avarage loss is:{}, l1_loss is {}, take time: {} seconds".format(global_step + 1, tatol_loss, tatol_loss1, eps_time_str))

        if (global_step + 1) % 10 == 0:
            raw2pixel = test_render_rays(pts, **render_kwargs_train)
            raw2pixel = raw2pixel.detach().cpu().numpy()
            raw2pixel = np.clip(raw2pixel, 0, 255)
            raw2pixel = raw2pixel * flag
            raw2pixel = np.rot90(raw2pixel)

            err = l2_error(raw2pixel, phantom) / 512.0
            tqdm.write(f"[TEST] Iter: {(global_step + 1)}; RMSE: {err}")
            log_string(f"[TEST] Iter: {(global_step + 1)}; RMSE: {err}")

            if err < best_rmse:
                best_rmse = err
                if (global_step + 1) == 50:
                    phantom_value = (phantom / vmax) * 255.0
                    imageio.imwrite(os.path.join(rec_dir, '0_gt.png'), (np.clip(phantom_value, 0, 255.0)).astype(np.uint8))
                phantom_value = (raw2pixel / vmax) * 255.0
                imageio.imwrite(os.path.join(result_dir, 'rec_result.png'),(np.clip(phantom_value, 0, 255.0)).astype(np.uint8))
                np.save(os.path.join(result_dir, 'best_result.npy'), raw2pixel)

                fig, subs = plt.subplots(1, 2, clear=True, num=1, figsize=(13, 5))
                fig.suptitle('RMSE: {}'.format(err))
                p0 = subs[0].imshow(phantom, aspect="auto", cmap='gray')
                subs[0].set_title("Gt phantom")
                plt.colorbar(p0, ax=subs[0])
                p1 = subs[1].imshow(raw2pixel, aspect="auto", cmap='gray', vmin=vmin, vmax=vmax)
                subs[1].set_title("Pred phantom")
                plt.colorbar(p1, ax=subs[1])
                fig.savefig(os.path.join(rec_dir, '{:03d}.png'.format(global_step+1)), bbox_inches="tight")

                # 保存权重
                path = os.path.join(result_dir, 'best.tar')
                torch.save({
                    'global_step': global_step + 1,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
                print('Saved checkpoints at', path)

if __name__=='__main__':
    parser = config_parser()
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda:{}'.format(args.gpu))
    else:
        device = torch.device('cpu')
    train(args)


