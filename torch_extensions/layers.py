import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple,Optional,Literal
from dataclasses import dataclass

class DyT(nn.Module):
    def __init__(self,dim,init_alpha=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1)*init_alpha) #Learnable scalar (shared across dim)
        self.gamma = nn.Parameter(torch.ones(dim))          #Learnable per-channel scale
        self.beta = nn.Parameter(torch.zeros(dim))          #Learnable per-channel shift
        
        
    def forward(self,x):
        """

        Args:
            x: Input Tensor of Shape [B,T,C] or [*,C] for generic transformer input.

        Returns:
            Tensor of same shape as input after applying DyT
        """
        x = torch.tanh(self.alpha*x)
        
        return self.gamma*x + self.beta
    
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Args:
        dim (int): Dimension of the input tensor.
        eps (float): Epsilon value for numerical stability. Defaults to 1e-6.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)