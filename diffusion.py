import torch
import torch.nn as nn
import torch.nn.functional as F

from UNet import UNet

class Diffusion(nn.Module):
    def __init__(self, device, noise_steps = 1000, beta_start = 1e-4, beta_end = 0.02, input_size = 28):
        super().__init__()
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.input_size = input_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    
    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def noise_input(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat[t])[:, None, None, None]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon
    
    def sample_timesteps(self, n):
        return torch.randint(low = 1, high = self.noise_steps, size = (n,))

    def sample(self, model, n):
        model.eval()
        with torch.no_grad():
            x = torch.rand((n, 1536, self.input_size, self.input_size)).to(self.device)
            for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.rand_like(x)
                else:
                    noise = torch.zeros_like(x)

                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x
    

if __name__ =='__main__':
    model = UNet(c_in=1536, c_out=1536, time_dim=28)
    diffusion = Diffusion(device='cpu', input_size=28)

    x = torch.randn(1, 1536, 28, 28)
    t = diffusion.sample_timesteps(x.shape[0])
    x_t, noise = diffusion.noise_input(x, t)

    print(x_t.shape, noise.shape)