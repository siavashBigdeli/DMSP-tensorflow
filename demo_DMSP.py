import numpy as np
import scipy.io as io
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

# The DMSP deblur function and the RGB filtering function (flipped convolution)
from DMSPDeblur import DMSPDeblur, filter_image
# The denoiser implementation
from DAE_model import denoiser

# Limit the GPU access
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"


# configure the tensorflow and instantiate a DAE
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.Session(config=config)

DAE = denoiser(sess)


# Load data
sigma_d = 255 * .01
matFile = io.loadmat('kernels.mat')
kernel = matFile['kernels'][0,0]
kernel = kernel / np.sum(kernel[:])
gt = np.array(Image.open('data/101085.jpg'), dtype='float32')

degraded = filter_image(gt, kernel)
noise = np.random.normal(0.0, sigma_d, degraded.shape).astype(np.float32)
degraded = degraded + noise

img_degraded = Image.fromarray(np.clip(degraded, 0, 255).astype(dtype=np.uint8))
img_degraded.save("data/degraded.png","png")


# non-blind deblurring demo
# run DMSP
params = {}
params['denoiser'] = DAE
params['sigma_dae'] = 11.0
params['num_iter'] = 300
params['mu'] = 0.9
params['alpha'] = 0.1
params['gt'] = gt # feed ground truth to monitor the PSNR at each iteration

restored = DMSPDeblur(degraded, kernel, sigma_d, params)

img_restored = Image.fromarray(np.clip(restored, 0, 255).astype(dtype=np.uint8))
img_restored.save("data/restored.png","png")

# noise-blind deblurring demo
# run DMSP noise-blind
params = {}
params['denoiser'] = DAE
params['sigma_dae'] = 11.0
params['num_iter'] = 300
params['mu'] = 0.9
params['alpha'] = 0.1
params['gt'] = gt

restored_nb = DMSPDeblur(degraded, kernel, -1, params)

img_restored_nb = Image.fromarray(np.clip(restored_nb, 0, 255).astype(dtype=np.uint8))
img_restored_nb.save("data/restored_noise_blind.png","png")