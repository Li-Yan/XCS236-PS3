import torch
from torch.nn import functional as F


def loss_nonsaturating_d(g, d, x_real, *, device):
    """
    Arguments:
    - g (codebase.network.Generator): The generator network
    - d (codebase.network.Discriminator): The discriminator network
      - Note that d outputs logits
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - device (torch.device): 'cpu' by default

    Returns:
    - d_loss (torch.Tensor): nonsaturating discriminator loss
    """
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)

    d_loss = None
    # You may find some or all of the below useful:
    #   - F.binary_cross_entropy_with_logits
    ### START CODE HERE ###
    d_loss = -1./batch_size * (F.logsigmoid(d(x_real)).sum() + torch.log(1. - torch.sigmoid(d(g(z)))).sum())
    return d_loss
    ### END CODE HERE ###
    raise NotImplementedError

def loss_nonsaturating_g(g, d, x_real, *, device):
    """
    Arguments:
    - g (codebase.network.Generator): The generator network
    - d (codebase.network.Discriminator): The discriminator network
      - Note that d outputs logits
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - device (torch.device): 'cpu' by default

    Returns:
    - g_loss (torch.Tensor): nonsaturating generator loss
    """
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)
    g_loss = None
    
    # You may find some or all of the below useful:
    #   - F.logsigmoid
    ### START CODE HERE ###
    g_loss = -1./batch_size * F.logsigmoid(d(g(z))).sum()
    return g_loss
    ### END CODE HERE ###
    raise NotImplementedError


def conditional_loss_nonsaturating_d(g, d, x_real, y_real, *, device):
    """
    Arguments:
    - g (codebase.network.ConditionalGenerator): The generator network
    - d (codebase.network.ConditionalDiscriminator): The discriminator network
      - Note that d outputs logits
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - y_real (torch.Tensor): training data labels (64)
    - device (torch.device): 'cpu' by default

    Returns:
    - d_loss (torch.Tensor): nonsaturating conditional discriminator loss
    """
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)
    d_loss = None

    ### START CODE HERE ###
    d_loss = -1./batch_size * (F.logsigmoid(d(x_real, y_real)).sum() + torch.log(1. - torch.sigmoid(d( g(z, y_real), y_real ))).sum())
    return d_loss
    ### END CODE HERE ###
    raise NotImplementedError


def conditional_loss_nonsaturating_g(g, d, x_real, y_real, *, device):
    """
    Arguments:
    - g (codebase.network.ConditionalGenerator): The generator network
    - d (codebase.network.ConditionalDiscriminator): The discriminator network
      - Note that d outputs logits
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - y_real (torch.Tensor): training data labels (64)
    - device (torch.device): 'cpu' by default

    Returns:
    - g_loss (torch.Tensor): nonsaturating conditional discriminator loss
    """
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)
    g_loss = None

    ### START CODE HERE ###
    g_loss = -1./batch_size * F.logsigmoid(d(g(z, y_real), y_real)).sum()
    return g_loss
    ### END CODE HERE ###
    raise NotImplementedError


def loss_wasserstein_gp_d(g, d, x_real, *, device):
    """
    Arguments:
    - g (codebase.network.Generator): The generator network
    - d (codebase.network.Discriminator): The discriminator network
      - Note that d outputs value of discriminator
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - device (torch.device): 'cpu' by default

    Returns:
    - d_loss (torch.Tensor): wasserstein discriminator loss
    """
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)
    d_loss = None

    # You may find some or all of the below useful:
    #   - torch.rand
    #   - torch.autograd.grad(..., create_graph=True)
    ### START CODE HERE ###
    alpha = torch.rand(batch_size, 1, 1, 1)
    rx = alpha * g(z) + (1. - alpha) * x_real
    rx_grad = torch.autograd.grad(d(rx).sum(), rx, create_graph=True)
    grad_norms = (rx_grad[0]**2).sum(dim=(1,2,3)).sqrt()

    d_loss = d(g(z)) - d(x_real)
    d_loss = d_loss + 10. * (grad_norms - 1.)**2
    d_loss = d_loss.mean()

    g_loss = -1. * d(g(z)).mean()
    return d_loss
    ### END CODE HERE ###
    raise NotImplementedError


def loss_wasserstein_gp_g(g, d, x_real, *, device):
    """
    Arguments:
    - g (codebase.network.Generator): The generator network
    - d (codebase.network.Discriminator): The discriminator network
      - Note that d outputs value of discriminator
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - device (torch.device): 'cpu' by default

    Returns:
    - g_loss (torch.Tensor): wasserstein generator loss
    """
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)
    g_loss = None
    
    ### START CODE HERE ###
    alpha = torch.rand(batch_size, 1, 1, 1)
    rx = alpha * g(z) + (1. - alpha) * x_real
    rx_grad = torch.autograd.grad(d(rx).sum(), rx, create_graph=True)
    grad_norms = (rx_grad[0]**2).sum(dim=(1,2,3)).sqrt()

    d_loss = d(g(z)) - d(x_real)
    d_loss = d_loss + 10. * (grad_norms - 1.)**2
    d_loss = d_loss.mean()

    g_loss = -1. * d(g(z)).mean()
    return g_loss
    ### END CODE HERE ###
    raise NotImplementedError