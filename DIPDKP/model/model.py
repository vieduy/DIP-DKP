import torch
import torch.nn.functional as F
import torchvision.transforms.functional as FF
import numpy as np
import sys
import tqdm
import os
from .networks import skip
from .SSIM import SSIM
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
sys.path.append('../')
from improved_diffusion import dist_util
from improved_diffusion.script_util import (
    create_model_and_diffusion,
)
import matplotlib.pyplot as plt
sys.path.append('../')
from util import save_final_kernel_png, get_noise, move2cpu, tensor2im01

class DIPFKP:
    '''
    # ------------------------------------------
    # (1) create model, loss and optimizer
    # ------------------------------------------
    '''

    def __init__(self, conf, lr, device=torch.device('cuda')):
        # setup
        self.conf = conf

        dist_util.setup_dist()
        self.model, self.diffusion = create_model_and_diffusion(
            image_size=16,
            num_channels=32,
            num_res_blocks=1,
            num_heads=1,
            num_heads_upsample=-1,
            attention_resolutions="4,2",
            dropout=0.5,
            learn_sigma=True,
            sigma_small=False,
            class_cond=False,
            diffusion_steps=1000,
            noise_schedule="cosine",
            timestep_respacing="ddim12",
            use_kl=False,
            predict_xstart=False,
            rescale_timesteps=True,
            rescale_learned_sigmas=True,
            use_checkpoint=False,
            use_scale_shift_norm=True,
        )

        if torch.cuda.is_available():
            self.model.load_state_dict(
            dist_util.load_state_dict('./model/ema_0.9999_100000.pt')
        )

        self.model.to(dist_util.dev())
        self.model.eval()
        print(sum(p.numel() for p in self.model.parameters() if p.requires_grad))

        # Acquire configuration
        self.lr = lr
        self.sf = conf.sf
        self.kernel_size = min(conf.sf * 4 + 3, 19)

        # DIP model
        _, C, H, W = self.lr.size()
        self.input_dip = get_noise(C, 'noise', (H * self.sf, W * self.sf)).to(device).detach()
        self.net_dip = skip(C, 3,
                            num_channels_down=[128, 128, 128, 128, 128],
                            num_channels_up=[128, 128, 128, 128, 128],
                            num_channels_skip=[16, 16, 16, 16, 16],
                            upsample_mode='bilinear',
                            need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU')
        self.net_dip = self.net_dip.to(device)
        # print(sum(p.numel() for p in self.net_dip.parameters() if p.requires_grad))
        self.optimizer_dip = torch.optim.Adam([{'params': self.net_dip.parameters()}], lr=conf.dip_lr)

        # normalizing flow as kernel prior
        if conf.model == 'DIPFKP':
            # initialze the kernel to be smooth is slightly better
            seed = 5
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.benchmark = True

            for p in self.model.parameters(): p.requires_grad = False

            self.kernel_code = torch.randn((1, 1, 16, 16), device=dist_util.dev())
            self.kernel_code.requires_grad = True
            self.optimizer_kp = SphericalOptimizer(self.kernel_size, torch.optim.Adam, [self.kernel_code],
                                                    lr=self.conf.kp_lr)

        # loss
        self.ssimloss = SSIM().to(device)
        self.mse = torch.nn.MSELoss().to(device)

        print('*' * 60 + '\nSTARTED {} on: {}...'.format(conf.model, conf.input_image_path))

    '''
    # ---------------------
    # (2) training
    # ---------------------
    '''

    def train(self):
        for iteration in tqdm.tqdm(range(self.conf.max_iters), ncols=60):
            iteration += 1

            self.optimizer_dip.zero_grad()
            if self.conf.model == 'DIPFKP':
                self.optimizer_kp.opt.zero_grad()
            else:
                self.optimizer_kp.zero_grad()

            '''
            # ---------------------
            # (2.1) forward
            # ---------------------
             '''

            # generate sr image
            sr = self.net_dip(self.input_dip)

            # generate kernel
            if self.conf.model == 'DIPFKP':
                alpha = 1e-6
                normalization = 0.16908
                model_kwargs = {}
                sample = self.diffusion.ddim_sample_loop(
                    self.model,
                    (1, 1, 16, 16),
                    self.kernel_code,
                    clip_denoised=True,
                    model_kwargs=model_kwargs,
                )
                sample = (sample + 1)*-7.0
                sample = ((torch.sigmoid(sample) - alpha) / (1 - 2 * alpha))
                sample = sample * normalization
                kernel = sample.contiguous()
                kernel = FF.resize(kernel, 19)
    

            # blur
            sr_pad = F.pad(sr, mode='circular',
                       pad=(self.kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2))
            out = F.conv2d(sr_pad, kernel.expand(3, -1, -1, -1), groups=3)

            # downscale
            out = out[:, :, 0::self.sf, 0::self.sf]
            # out = out[:, :, :-1, :-1]


            '''
            # ---------------------
            # (2.2) backward
            # ---------------------
             '''
            # freeze kernel estimation, so that DIP can train first to learn a meaningful image
            if iteration <= 75:
                self.kernel_code.requires_grad = False
            else:
                self.kernel_code.requires_grad = True

            # first use SSIM because it helps the model converge faster
            if iteration <= 80:
                loss = 1 - self.ssimloss(out, self.lr)
            else:
                loss = self.mse(out, self.lr)

            loss.backward()
            self.optimizer_dip.step()
            self.optimizer_kp.step()


            if (iteration % 100 == 0 or iteration == 1):
                plt.imsave(os.path.join(self.conf.output_dir_path, '{}_{}.png'.format(self.conf.img_name, iteration)),
                                        tensor2im01(sr), vmin=0, vmax=1., dpi=1)
                # save_final_kernel_png(move2cpu(kernel.squeeze()), self.conf, self.conf.kernel_gt, iteration)
                print('\n Iter {}, loss: {}'.format(iteration, loss.data))


            del out
            del sr_pad
            torch.cuda.empty_cache()
            if iteration != 300:
                del kernel

        kernel = move2cpu(kernel.squeeze())
        # self.conf.kernel_gt = np.resize(self.conf.kernel_gt, (16, 16))
        save_final_kernel_png(kernel, self.conf, self.conf.kernel_gt)

        if self.conf.verbose:
            print('{} estimation complete! (see --{}-- folder)\n'.format(self.conf.model,
                                                                         self.conf.output_dir_path) + '*' * 60 + '\n\n')

        return kernel, sr


class SphericalOptimizer(torch.optim.Optimizer):
    ''' spherical optimizer, optimizer on the sphere of the latent space'''

    def __init__(self, kernel_size, optimizer, params, **kwargs):
        self.opt = optimizer(params, **kwargs)
        self.params = params
        with torch.no_grad():
            # in practice, setting the radii as kernel_size-1 is slightly better
            self.radii = {param: torch.ones([1, 1, 1]).to(param.device) * (kernel_size - 1) for param in params}
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = self.opt.step(closure)
        for param in self.params:
            param.data.div_((param.pow(2).sum(tuple(range(2, param.ndim)), keepdim=True) + 1e-9).sqrt())
            param.mul_(self.radii[param])
        return loss