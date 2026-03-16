import torch


class BaseFlowMatchingHelper:
    """
    Base helper for computing x_t and u_t, given target x_1 and noise x_0
    ref:  Flow matching for generative modeling, Lipman
    """
    def __init__(self, sigma=1e-5):
        self.sigma = sigma

    def compute_mu_t(self, x1, t):
        """Eq. 20 in the paper"""
        return t * x1

    def compute_sigma_t(self, t):
        """Eq. 20 in the paper"""
        return 1 - (1 - self.sigma) * t

    def sample_x_t(self, x0, x1, t):
        mu_t = self.compute_mu_t(x1, t)
        sigma_t = self.compute_sigma_t(t)
        return mu_t + sigma_t * x0

    def compute_u_t(self, x0, x1):
        """
        Eq. 21 in the paper:  u_t = (x1 - (1 - self.sigma) * xt) / (1 - (1 - self.sigma) * t)
        it equals to the following:
        """
        u_t =  x1 - (1 - self.sigma) * x0
        return u_t

    def compute_xt_ut(self, x1, t=None):
        x0 = torch.randn_like(x1, device=x1.device)
        if t is None:
            t = torch.rand(x1.size(0), dtype=x1.dtype, device=x1.device)
        times = t
        t = t.reshape(-1, *([1] * (x1.dim() - 1)))
        xt = self.sample_x_t(x0, x1, t)
        ut = self.compute_u_t(x0, x1)
        return xt, ut, times



if __name__ == "__main__":
    x1 = torch.rand(4, 10, 16)
    helper = BaseFlowMatchingHelper(sigma=1e-5)
    xt, ut, t = helper(x1)
