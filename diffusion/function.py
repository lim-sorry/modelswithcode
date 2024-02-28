import torch

class DDIM:
    def __init__(self, b_s, b_e, t, device) -> None:
        # Beta schedule for timestep T
        self.b = torch.linspace(b_s, b_e, t).to(device)
        self.a = 1 - self.b
        self.a_bar = torch.cumprod(self.a, 0)

        self.sqrt_a_bar = torch.sqrt(self.a_bar)
        self.sqrt_one_minus_a_bar = torch.sqrt(1 - self.a_bar)

        self.inv_sqrt_a = 1 / torch.sqrt(self.a)
        self.u_e = self.b / self.sqrt_one_minus_a_bar
        self.b_hat = torch.sqrt((1 - torch.roll(self.a_bar, 1)) / (1 - self.a_bar) * self.b)
        

    def sample_noise_image(self, image:torch.Tensor, t:torch.Tensor) -> torch.Tensor:
        t = t.int()
        
        noise = torch.randn_like(image)
        image = self.sqrt_a_bar[t, None, None, None] * image + self.sqrt_one_minus_a_bar[t, None, None, None] * noise
        image = torch.clamp(image, -1.0, 1.0)
        return image, noise
    

    def denoise_image(self, image:torch.Tensor, noise:torch.Tensor, t:torch.Tensor):
        t = t.int()
        u_t = self.inv_sqrt_a[t, None, None, None] * (image - self.u_e[t, None, None, None] * noise)
        
        if t[0].item() == 0:
            return u_t
        
        s_t = torch.sqrt(self.b[t, None, None, None]) * torch.randn_like(image)
        return u_t + s_t
        
