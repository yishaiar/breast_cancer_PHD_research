import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch import max,min


class RBM(nn.Module):
    r"""Restricted Boltzmann Machine.

    Args:
        n_vis (int, optional): The size of visible layer. Defaults to 784.
        n_hid (int, optional): The size of hidden layer. Defaults to 128.
        k (int, optional): The number of Gibbs sampling. Defaults to 1.
    """

    def __init__(self, n_vis=784, n_hid=128,output_shape = 784, k=1):
        """Create a RBM."""
        super(RBM, self).__init__()
        self.v = nn.Parameter(torch.randn( n_vis))
        self.h = nn.Parameter(torch.randn( n_hid))
        self.W = nn.Parameter(torch.randn(n_hid, n_vis))
        self.k = k
        self.output_shape = output_shape
        self.n_vis = n_vis
    def visible_to_hidden(self, v):
        r"""Conditional sampling a hidden variable given a visible variable.

        Args:
            v (Tensor): The visible variable.

        Returns:
            Tensor: The hidden variable.

        """
        p = torch.sigmoid(F.linear(v, self.W, self.h))
        
        return p.bernoulli()

    def hidden_to_visible(self, h):
        r"""Conditional sampling a visible variable given a hidden variable.

        Args:
            h (Tendor): The hidden variable.

        Returns:
            Tensor: The visible variable.
            
        torch.nn.functional.linear(input, weight, bias=None) → Tensor
        Applies a linear transformation to the incoming data: 
        𝑦 = X * A_t + b

 

        """
        p = torch.sigmoid(F.linear(h, self.W.t(), self.v))
        #bernoulli() returns a binary value
        # by Drawing binary random numbers (0 or 1) from a Bernoulli distribution.
        

        return p.bernoulli()

    def free_energy(self, v):
        r"""Free energy function.

        .. math::
            \begin{align}
                F(x) &= -\log \sum_h \exp (-E(x, h)) \\
                &= -a^\top x - \sum_j \log (1 + \exp(W^{\top}_jx + b_j))\,.
            \end{align}

        Args:
            v (Tensor): The visible variable.

        Returns:
            FloatTensor: The free energy value.

        """
        v_term = torch.matmul(v, self.v.t())
        w_x_h = F.linear(v, self.W, self.h)
        h_term = torch.sum(F.softplus(w_x_h), dim=1)
        return torch.mean(-h_term - v_term)

    def forward(self, v):
        r"""Compute the real and generated examples.

        Args:
            v (Tensor): The visible variable (layer).
            Each node in this layer represents one feature of the input

        Returns:
            (Tensor, Tensor): The real and generagted variables.

        """
        # s = v.size()
        v = v.view(-1,self.n_vis)
        h = self.visible_to_hidden(v)
        for _ in range(self.k):
            v_gibb = self.hidden_to_visible(h)
            h = self.visible_to_hidden(v_gibb)
        return v_gibb.view(-1,*self.output_shape)
