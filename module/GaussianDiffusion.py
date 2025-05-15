import torch
import torch.nn as nn
import numpy as np
from torch_geometric.utils import to_dense_adj


class GaussianDiffusion(nn.Module):
    def __init__(self, noise_scale, noise_min, noise_max, steps, beta_fixed=True):
        super(GaussianDiffusion, self).__init__()

        self.noise_scale = noise_scale
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.steps = steps

        if noise_scale != 0:
            self.betas = torch.tensor(self.get_betas(), dtype=torch.float64)
            if beta_fixed:
                self.betas[0] = 0.0001

            self.calculate_for_diffusion()

    def get_betas(self):
        start = self.noise_scale * self.noise_min
        end = self.noise_scale * self.noise_max
        variance = np.linspace(start, end, self.steps, dtype=np.float64)
        alpha_bar = 1 - variance
        betas = []
        betas.append(1 - alpha_bar[0])
        for i in range(1, self.steps):
            betas.append(min(1 - alpha_bar[i] / alpha_bar[i - 1], 0.999))
        return np.array(betas)

    def calculate_for_diffusion(self):
        alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0])])

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (
                self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]]))
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = (
                    (1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod))

    def p_sample(self, model, x_start, steps, edge_index, y, sampling_noise=False):
        if steps == 0:
            x_t = x_start
        else:
            t = torch.tensor([steps - 1] * x_start.shape[0])
            x_t = self.q_sample(x_start, t)

        indices = list(range(self.steps))[::-1]

        for i in indices:
            t = torch.tensor([i] * x_t.shape[0])
            model_mean, model_log_variance = self.p_mean_variance(model, x_t, t, edge_index, y)
            if sampling_noise:
                noise = torch.randn_like(x_t)
                nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
                x_t = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
            else:
                x_t = model_mean
        return x_t

    def q_sample(self, x_start, ts, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return self._extract_into_tensor(self.sqrt_alphas_cumprod, ts,
                                         x_start.shape) * x_start + self._extract_into_tensor(
            self.sqrt_one_minus_alphas_cumprod, ts, x_start.shape) * noise

    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        arr = arr
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)

    def p_mean_variance(self, model, x, t, edge_index, y):

        model_output = model(x, t, edge_index, y)

        model_variance = self.posterior_variance
        model_log_variance = self.posterior_log_variance_clipped

        model_variance = self._extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)

        model_mean = (self._extract_into_tensor(self.posterior_mean_coef1, t,
                                                x.shape) * model_output + self._extract_into_tensor(
            self.posterior_mean_coef2, t, x.shape) * x)

        return model_mean, model_log_variance

    def training_losses(self, model, x_start, edge_index, y):
        batch_size = x_start.size(0)

        ts = torch.randint(0, self.steps, (batch_size,)).long()
        noise = torch.randn_like(x_start)
        if self.noise_scale != 0:
            x_t = self.q_sample(x_start, ts, noise)
        else:
            x_t = x_start

        model_output = model(x_t, ts, edge_index, y)

        mse = self.mean_flat((x_start - model_output) ** 2)

        weight = self.SNR(ts - 1) - self.SNR(ts)
        weight = torch.where((ts == 0), 1.0, weight)

        diff_loss = weight * mse

        return diff_loss

    def mean_flat(self, tensor):
        return tensor.mean(dim=list(range(1, len(tensor.shape))))

    def SNR(self, t):
        self.alphas_cumprod = self.alphas_cumprod
        return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])