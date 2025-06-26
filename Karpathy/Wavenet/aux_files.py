import torch
import torch.nn
import matplotlib.pyplot as plt

class Linear:

    def __init__(self, fan_in, fan_out, random_seed = 46678587847, bias = True):
        # pytorch expects a seed from torch.Generator(), not an integer directly!
        gen = torch.Generator().manual_seed(random_seed) 
        self.weight = torch.randn((fan_in, fan_out), generator=gen) * 1/fan_in**0.5
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
        # if self.training:
        #     xmean = x.mean(dim = 0, keepdim = True)
        #     xvar = x.var(dim = 0, keepdim = True)

        # After fixing bug -- we depart from pytorch.nn.Batchnorm1d API
        if self.training:
            if x.ndim == 2:
                dim = 0
            elif x.ndim == 3:
                dim = (0,1)
            xmean = x.mean(dim, keepdim=True) # batch mean
            xvar = x.var(dim, keepdim=True) # batch variance

        else: 
            xmean = self.running_mean
            xvar = self.running_var
        # apply to data
        xhat = (x-xmean)/torch.sqrt(xvar + self.eps)

        self.out = self.gamma * xhat + self.beta # bngain * xhat + bnbias 
 
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
    
class Embedding:

    def __init__(self, num_embeddings, embed_dim):
        self.weight = torch.randn((num_embeddings,embed_dim))
        
    def __call__(self, Xb):
        self.out = self.weight[Xb]
        return self.out
    
    def parameters(self):
        return [self.weight]


class Flatten:

    def __call__(self, x):
        self.out = x.view(x.shape[0], -1)
        return self.out
    
    def parameters(self):
        return []
    
#-----------------------------------------

class Sequential:

    def __init__(self, layers):
        self.layers = layers
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out
    
    def parameters(self):
        #get all parameters and stretch them out into a list. 
        return [p for layer in self.layers for p in layer.parameters()]
    
# --------------------------------------

class FlattenConsecutive:
  
  def __init__(self, n):
    # n = number of consecutive chars to be used as context/ grouped
    self.n = n
    
  def __call__(self, x):
    B, T, C = x.shape
    x = x.view(B, T//self.n, C*self.n)
    if x.shape[1] == 1:
      # to counter spurious dimension in case 3D is actually 2D - check torch.squeeze documentation for clarity. 
      x = x.squeeze(1)
    self.out = x
    return self.out
  
  def parameters(self):
    return []