import torch
import torch.nn
import matplotlib.pyplot as plt

class Linear:

    def __init__(self, fan_in, fan_out, bias = True):
        self.weight = torch.randn((fan_in, fan_out), generator=g) * 1/fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None
    
    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    
    def parameters(self):
        params = [self.weight] + ([] if self.bias is None else [self.bias])
        return params
    

class BatchNorm1d: 

    def __init__(self, size, eps = 1e-5, momentum = 0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # parameters (trained with backprop)
        self.gamma = torch.ones(size)
        self.beta = torch.zeros(size)
        # buffers (trained with a running 'momentum update')
        self.running_mean = torch.zeros(size)
        self.running_var = torch.ones(size)
    
    def __call__(self, x):
        if self.training:
            xmean = x.mean(dim = 0, keepdim = True)
            xvar = x.var(dim = 0, keepdim = True)

        else: 
            xmean = self.running_mean
            xvar = self.running_var
        # apply to data
        xhat = (x-xmean)/torch.sqrt(xvar + self.eps)

        self.out = self.gamma * xhat + self.beta # gain * xhat + bias 
 
        if self.training:
            with torch.no_grad():
                self.running_mean = self.momentum * xmean + (1 - self.momentum) * self.running_mean
                self.running_var = self.momentum * xvar + (1 - self.momentum) * self.running_var

        return self.out
    
    def parameters(self):
        return [self.gamma , self.beta]

class Tanh:

    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    def parameters(self):
        return []