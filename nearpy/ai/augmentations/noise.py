import numpy as np

class GaussianNoise:
    """Add Gaussian noise to data"""
    def __init__(self, std=0.1, mean=0.0):
        self.std = std
        self.mean = mean
    
    def __call__(self, x):
        noise = np.random.normal(self.mean, self.std, x.shape)
        return x + noise


class UniformNoise:
    """Add uniform noise to data"""
    def __init__(self, low=-0.1, high=0.1):
        self.low = low
        self.high = high
    
    def __call__(self, x):
        noise = np.random.uniform(self.low, self.high, x.shape)
        return x + noise


class SaltPepperNoise:
    """Add salt and pepper noise (random values set to min/max)"""
    def __init__(self, prob=0.05):
        self.prob = prob
    
    def __call__(self, x):
        x_noisy = x.copy()
        mask = np.random.rand(*x.shape) < self.prob
        # Salt (set to max)
        salt_mask = mask & (np.random.rand(*x.shape) < 0.5)
        x_noisy[salt_mask] = np.max(x)
        # Pepper (set to min)
        pepper_mask = mask & ~salt_mask
        x_noisy[pepper_mask] = np.min(x)
        return x_noisy


class PoissonNoise:
    """Add Poisson noise (useful for count-based data)"""
    def __init__(self, lam=1.0):
        self.lam = lam
    
    def __call__(self, x):
        # Scale data to positive range for Poisson
        x_min = np.min(x)
        x_shifted = x - x_min + self.lam
        noise = np.random.poisson(lam=self.lam, size=x.shape)
        return x_shifted + noise