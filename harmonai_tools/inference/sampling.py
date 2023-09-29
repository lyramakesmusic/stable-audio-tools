import torch
import math
from tqdm import trange

import k_diffusion as K

# Define the noise schedule and sampling loop
def get_alphas_sigmas(t):
    """Returns the scaling factors for the clean image (alpha) and for the
    noise (sigma), given a timestep."""
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)

def alpha_sigma_to_t(alpha, sigma):
    """Returns a timestep, given the scaling factors for the clean image and for
    the noise."""
    return torch.atan2(sigma, alpha) / math.pi * 2

@torch.no_grad()
def sample(model, x, steps, eta, **extra_args):
    """Draws samples from a model given starting noise. v-diffusion"""
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    t = torch.linspace(1, 0, steps + 1)[:-1]

    alphas, sigmas = get_alphas_sigmas(t)

    # The sampling loop
    for i in trange(steps):

        # Get the model output (v, the predicted velocity)
        with torch.cuda.amp.autocast():
            v = model(x, ts * t[i], **extra_args).float()

        # Predict the noise and the denoised image
        pred = x * alphas[i] - v * sigmas[i]
        eps = x * sigmas[i] + v * alphas[i]

        # If we are not on the last timestep, compute the noisy image for the
        # next timestep.
        if i < steps - 1:
            # If eta > 0, adjust the scaling factor for the predicted noise
            # downward according to the amount of additional noise to add
            ddim_sigma = eta * (sigmas[i + 1]**2 / sigmas[i]**2).sqrt() * \
                (1 - alphas[i]**2 / alphas[i + 1]**2).sqrt()
            adjusted_sigma = (sigmas[i + 1]**2 - ddim_sigma**2).sqrt()

            # Recombine the predicted noise and predicted denoised image in the
            # correct proportions for the next step
            x = pred * alphas[i + 1] + eps * adjusted_sigma

            # Add the correct amount of fresh noise
            if eta:
                x += torch.randn_like(x) * ddim_sigma

    # If we are on the last timestep, output the denoised image
    return pred

# Soft mask inpainting is just shrinking hard (binary) mask inpainting
# Given a float-valued soft mask (values between 0 and 1), get the binary mask for this particular step
def get_bmask(i, steps, mask):
    strength = (i+1)/(steps)
    # convert to binary mask
    bmask = torch.where(mask<=strength,1,0)
    return bmask

# Uses k-diffusion from https://github.com/crowsonkb/k-diffusion
# For sampling, set both init_audio and mask to None
# For variations, set init_audio 
# For inpainting, set both init_audio & mask 
#def sample_k(model_fn, noise, init_audio=None, init_noise_level=0.1, mask=None, steps=100, sampler_type="dpmpp-2m-sde", sigma_max=80, sigma_min=0.5, rho=1.0, device="cuda", callback=None, **extra_args):

def sample_k(
        model_fn, 
        noise, 
        init_audio=None,
        mask=None,
        steps=100, 
        sampler_type="dpmpp-2m-sde", 
        sigma_min=0.5, 
        sigma_max=50, 
        rho=1.0, device="cuda", 
        callback=None, 
        init_data=None,
        **extra_args
    ):

    denoiser = K.external.VDenoiser(model_fn)

    # Make the list of sigmas. Sigma values are scalars related to the amount of noise each denoising step has
    sigmas = K.sampling.get_sigmas_polyexponential(steps, sigma_min, sigma_max, rho, device=device)
    # Scale the initial noise by sigma 
    noise = noise * sigmas[0]

    wrapped_callback = callback

    if mask is None and init_audio is not None:
        # VARIATION (no inpainting)
        # set the initial latent to the init_audio, and noise it with initial sigma
        x = init_audio + noise 
    elif mask is not None and init_audio is not None:
        # INPAINTING
        bmask = get_bmask(0, steps, mask)
        # initial noising
        input_noised = init_audio + noise
        # set the initial latent to a mix of init_audio and noise, based on step 0's binary mask
        x = input_noised * bmask + noise * (1-bmask)
        # define the inpainting callback function (Note: side effects, it mutates x)
        # See https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/sampling.py#L596C13-L596C105
        # callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        # This is called immediately after `denoised = model(x, sigmas[i] * s_in, **extra_args)`
        def inpainting_callback(args):
            i = args["i"]
            x = args["x"]
            sigma = args["sigma"]
            #denoised = args["denoised"]
            # noise the init_audio input with this step's appropriate amount of noise
            input_noised = init_audio + torch.randn_like(init_audio) * sigma
            # shrinking hard mask
            bmask = get_bmask(i, steps, mask)
            # mix input_noise with x, using binary mask
            new_x = input_noised * bmask + x * (1-bmask)
            # mutate x
            x[:,:,:] = new_x[:,:,:]
        # wrap together the inpainting callback and the user-submitted callback. 
        if callback is not None: 
            wrapped_callback = lambda args: (inpainting_callback(args), callback(args))
    else:
        # SAMPLING
        # set the initial latent to noise
        x = noise


    with torch.cuda.amp.autocast():
        if sampler_type == "k-heun":
            return K.sampling.sample_heun(denoiser, x, sigmas, disable=False, callback=wrapped_callback, extra_args=extra_args)
        elif sampler_type == "k-lms":
            return K.sampling.sample_lms(denoiser, x, sigmas, disable=False, callback=wrapped_callback, extra_args=extra_args)
        elif sampler_type == "k-dpmpp-2s-ancestral":
            return K.sampling.sample_dpmpp_2s_ancestral(denoiser, x, sigmas, disable=False, callback=wrapped_callback, extra_args=extra_args)
        elif sampler_type == "k-dpm-2":
            return K.sampling.sample_dpm_2(denoiser, x, sigmas, disable=False, callback=wrapped_callback, extra_args=extra_args)
        elif sampler_type == "k-dpm-fast":
            return K.sampling.sample_dpm_fast(denoiser, x, sigma_min, sigma_max, steps, disable=False, callback=wrapped_callback, extra_args=extra_args)
        elif sampler_type == "k-dpm-adaptive":
            return K.sampling.sample_dpm_adaptive(denoiser, x, sigma_min, sigma_max, rtol=0.01, atol=0.01, disable=False, callback=wrapped_callback, extra_args=extra_args)
        elif sampler_type == "dpmpp-2m-sde":
            return K.sampling.sample_dpmpp_2m_sde(denoiser, x, sigmas, disable=False, callback=wrapped_callback, extra_args=extra_args)

