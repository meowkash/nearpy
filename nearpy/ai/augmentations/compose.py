'''
Usage: 
- Single transform
    self.transform = GaussianNoise(std=0.05)

- Composed transforms
    self.transform = Compose([
        GaussianNoise(std=0.05),
        UniformNoise(low=-0.02, high=0.02)
    ])
'''

class Compose:
    """Chain multiple transforms together"""
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x