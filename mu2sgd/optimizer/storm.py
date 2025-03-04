import torch
from torch.optim.optimizer import Optimizer, required


class STORM(Optimizer):
    """
    A custom implementation of the STORM optimizer, which is designed to handle stochastic gradient updates
    with momentum and weight decay. It supports a gradient estimator and optionally an alpha-based weighting scheme.

    Attributes:
    - iter (int): Tracks the number of iterations completed.
    - use_beta_t (bool): Whether to use a decaying momentum factor at the iterations number for updates.
    """

    def __init__(self, params, lr=required, weight_decay=0., momentum=0.9, use_beta_t=False):
        """
        Initializes the STORM optimizer.

        Args:
        - params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        - lr (float): Learning rate (required).
        - weight_decay (float, optional): L2 regularization coefficient. Default is 0.0.
        - momentum (float, optional): Momentum factor for gradient updates. Default is 0.9.
        - use_beta_t (bool, optional): Use decaying momentum factor at the iterations number if True. Default is False.
        """
        defaults = dict(lr=lr, beta=momentum, weight_decay=weight_decay)
        super(STORM, self).__init__(params, defaults)

        # Initialize optimizer state for each parameter
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['d_t'] = torch.full_like(p.data, 0.)
                state['current_grad'] = torch.full_like(p.data, 0.)
                state['correction_grad'] = torch.full_like(p.data, 0.)

        self.iter = 0
        self.use_beta_t = use_beta_t

    def __setstate__(self, state):
        """
        Set the state of the optimizer.

        This method overrides the base `__setstate__` method to ensure the optimizer's state
        is properly restored after being loaded from a checkpoint.

        Args:
        - state (dict): The state dictionary to restore.
        """
        super(STORM, self).__setstate__(state)

    def compute_estimator(self):
        """
        Computes the gradient estimator for the STORM optimizer.

        This method updates the internal gradient estimate (`d_t`) based on the current gradient,
        the correction gradient, and the momentum or alpha-based weighting scheme.
        """
        self.iter += 1
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

                # Update the correction gradient
                state = self.state[p]
                state['correction_grad'] = p_grad.detach()

        # Update parameters using the gradient estimator
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue

                # Apply gradient-based update
                state = self.state[p]
                p.data.add_(state['d_t'], alpha=-lr)

        return loss
