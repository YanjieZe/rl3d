import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np
import os
import json
import random
import subprocess
import platform
from datetime import datetime
from copy import deepcopy
from multiprocessing import Process
from termcolor import colored
import time
import cv2

class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def cat(x, y, axis=0):
    return torch.cat([x, y], axis=0)


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    


def write_info(args, fp):
    data = {
        'host': platform.node(),
        'cwd': os.getcwd(),
        'timestamp': str(datetime.now()),
        'args': vars(args)
    }
    with open(fp, 'w') as f:
        json.dump(data, f, indent=4, separators=(',', ': '))


def load_config(key=None):
    path = os.path.join('setup', 'config.cfg')
    with open(path) as f:
        data = json.load(f)
    if key is not None:
        return data[key]
    return data


def make_dir(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path


def prefill_memory(capacity, obs_shape):
    obses = []
    if len(obs_shape) > 1:
        c, h, w = obs_shape
        for _ in range(capacity):
            frame = np.ones((c, h, w), dtype=np.uint8)
            obses.append(frame)
    else:
        for _ in range(capacity):
            obses.append(np.ones(obs_shape[0], dtype=np.float32))

    return obses



class LazyFrames(object):
    def __init__(self, frames, extremely_lazy=True):
        self._frames = frames
        self._extremely_lazy = extremely_lazy
        self._out = None

    @property
    def frames(self):
        return self._frames

    def _force(self):
        if self._extremely_lazy:
            return np.concatenate(self._frames, axis=0)
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=0)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        if self._extremely_lazy:
            return len(self._frames)
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        if self.extremely_lazy:
            return len(self._frames)
        frames = self._force()
        return frames.shape[0] // 3

    def frame(self, i):
        return self._force()[i * 3:(i + 1) * 3]


def count_parameters(net, as_int=False):
    """Returns number of params in network"""
    count = sum(p.numel() for p in net.parameters())
    if as_int:
        return count
    return f'{count:,}'


def save_obs(obs, fname='obs', resize_factor=None):
    assert obs.ndim == 3, 'expected observation of shape (C, H, W)'
    if isinstance(obs, torch.Tensor):
        obs = obs.detach().cpu()
    else:
        obs = torch.FloatTensor(obs)
    c, h, w = obs.shape
    if resize_factor is not None:
        obs = torchvision.transforms.functional.resize(obs, size=(h * resize_factor, w * resize_factor))
    if c == 3:
        torchvision.utils.save_image(obs / 255., fname + '.png')
    elif c == 9:
        grid = torch.stack([obs[i * 3:(i + 1) * 3] for i in range(3)], dim=0)
        grid = torchvision.utils.make_grid(grid, nrow=3)
        torchvision.utils.save_image(grid / 255., fname + '.png')
    else:
        raise NotImplementedError('save_obs does not support other number of channels than 3 or 9')


def parallel(fn, args, wait=10):
	"""Executes function multiple times in parallel, using individual seeds (given in args.seed)"""
	assert not isinstance(fn, (list, tuple)), 'fn must be a function, not a list or tuple'
	assert args.seed is not None, 'No seed(s) given'
	seeds = args.seed
	if not isinstance(seeds, (list, tuple)):
		return fn(args)
	proc = []
	for seed in seeds:
		_args = deepcopy(args)
		_args.seed = seed
		p = Process(target=fn, args=(_args,))
		p.start()
		proc.append(p)
		print(colored(f'Started process {p.pid} with seed {seed}', 'green', attrs=['bold']))
		time.sleep(wait) # ensure that seed has been set in child process
	for p in proc:
		p.join()
	while len(proc) > 0:
		time.sleep(wait)
		for p in proc:
			if not p.is_alive():
				p.terminate()
				proc.remove(p)
				print(f'One of the child processes have terminated.')
	exit(0)

def save_image(obs, fname):
    torchvision.utils.save_image(obs / 255., fname + '.png')


class PSNR:
    
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]

    link: https://github.com/bonlime/pytorch-tools/blob/master/pytorch_tools/metrics/psnr.py"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(255.0 / torch.sqrt(mse))


class SSIM:
    """Structure Similarity
    img1, img2: [0, 255]

    link: https://github.com/bonlime/pytorch-tools/blob/master/pytorch_tools/metrics/psnr.py"""

    def __init__(self):
        self.name = "SSIM"

    def __call__(self, img1, img2):
        if not img1.shape == img2.shape:
            raise ValueError("Input images must have the same dimensions.")
        if img1.ndim == 2:  # Grey or Y-channel image
            return self._ssim(img1, img2)
        elif img1.ndim == 3:
            if img1.shape[2] == 3:
                ssims = []
                for i in range(3):
                    ssims.append(self._ssim(img1, img2))
                return np.array(ssims).mean()
            elif img1.shape[2] == 1:
                return self._ssim(np.squeeze(img1), np.squeeze(img2))
            else:
                raise ValueError("Input images must have 1 or 3 channels.")
        else:
            raise ValueError("Wrong input image dimensions.")

    def _ssim(self, img1, img2):
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        # img1 = img1.astype(np.float64)
        # img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()