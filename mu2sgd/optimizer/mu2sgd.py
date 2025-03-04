import torch
from torch.optim.optimizer import Optimizer, required
import copy
from ..utils import proj


class Mu2SGD(Optimizer):
    """
    A custom implementation of the μ²-SGD optimizer.

    Attributes:
    - gamma (float): Controls interpolation between parameter values during updates.
    - projection_radius (float or None): Radius for projecting parameter updates to a constrained set.
    - iter (int): Tracks the number of iterations completed.
    - sum_iter (int): Cumulative sum of iteration indices, used in weighted parameter updates.
    - use_alpha_t (bool): Whether to use alpha_t=t in the iterations number for updates.
    - use_beta_t (bool): Whether to use decaying beta in the iterations number for updates.
    - w (list): Deep copies of parameter groups for maintaining intermediate states during updates.
    """

    def __init__(self, params, lr=required, weight_decay=0., momentum=0.9, gamma=0.9, use_alpha_t=False,
                 use_beta_t=False, projection_radius=None):
        """
        Initializes the μ²-SGD optimizer.

        Args:
        - params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        - lr (float): Learning rate (required).
        - weight_decay (float, optional): L2 regularization coefficient. Default is 0.0.
        - momentum (float, optional): Momentum factor for gradient updates. Default is 0.9.
        - gamma (float, optional): Interpolation factor for parameter updates. Default is 0.9.
        - use_alpha_t (bool, optional): Use alpha_t=t in the iterations number if True. Default is False.
        - use_beta_t (bool, optional): Use decaying beta in the iterations number if True. Default is False.
        - projection_radius (float or None, optional): Radius for parameter projection. Default is None.
        """
        defaults = dict(lr=lr, beta=momentum, weight_decay=weight_decay, gamma=gamma)
        super(Mu2SGD, self).__init__(params, defaults)

        # Initialize optimizer state for each parameter
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['d_t'] = torch.full_like(p.data, 0.)
                state['current_grad'] = torch.full_like(p.data, 0.)
                state['correction_grad'] = torch.full_like(p.data, 0.)

        self.gamma = gamma
        self.projection_radius = projection_radius
        self.iter = 0
        self.sum_iter = 0
        self.use_alpha_t = use_alpha_t
        self.use_beta_t = use_beta_t

        # Create a deep copy of parameter groups to maintain intermediate states
        self.w = []
        for group in self.param_groups:
            cloned_group = copy.deepcopy(group)
            for i, p in enumerate(group['params']):
                cloned_group['params'][i] = p.clone().detach().requires_grad_(p.requires_grad)
            self.w.append(cloned_group)

    def __setstate__(self, state):
        """
        Set the state of the optimizer.
        Overrides the base `__setstate__` to restore state information.
        """
        super(Mu2SGD, self).__setstate__(state)

    def compute_estimator(self):
        """
        Compute the gradient estimator for the μ²-SGD optimizer.

        """
        self.iter += 1
        self.sum_iter += self.iter

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                p_grad = p.grad.data
                if weight_decay != 0:
                    p_grad.add_(weight_decay, p.data)

                # Update state with current gradient and compute the gradient estimator
                state = self.state[p]
                state['current_grad'] = p_grad.detach()
                if self.use_beta_t:
                    beta = 1 / self.iter
                    state['d_t'] = (state['current_grad'] + (1. - beta) * (
                            state['d_t'] - state['correction_grad'])).detach()
                else:
                    state['d_t'] = (state['current_grad'] + (1. - group['beta']) * (
                            state['d_t'] - state['correction_grad'])).detach()

    def step(self, closure=None):
        """
        Performs a single optimization step (parameter update).

        Args:
        - closure (callable, optional): A closure that reevaluates the model and returns the loss.

        Returns:
        - loss (float or None): The loss value if the closure is provided; otherwise, None.
        """
        loss = None
        if closure is not None:
            loss = closure()

        # Update correction gradients
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                p_grad = p.grad.data
                if weight_decay != 0:
                    p_grad.add_(weight_decay, p.data)

                state = self.state[p]
                state['correction_grad'] = p_grad.detach()

        # Update parameters using the gradient estimator
        for group, group_w in zip(self.param_groups, self.w):
            lr = group['lr']
            weight_decay = group['weight_decay']
            for p, pw in zip(group['params'], group_w['params']):
                if p.grad is None:
                    continue
                p_grad = p.grad.data
                if weight_decay != 0:
                    p_grad.add_(weight_decay, p.data)

                # Apply gradient-based update
                state = self.state[p]
                pw.data.add_(state['d_t'], alpha=-lr)

                # Apply projection if specified
                if self.projection_radius:
                    pw.data = proj(pw.data, self.projection_radius)

                # Update parameters using alpha or gamma interpolation
                if self.use_alpha_t:
                    a = self.iter / self.sum_iter
                    b = (self.sum_iter - self.iter) / self.sum_iter
                    p.data = a * pw.data + b * p.data
                else:
                    p.data = self.gamma * pw.data + (1 - self.gamma) * p.data

        return loss
