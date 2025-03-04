from torch.optim import Optimizer
import copy
from ..utils import proj


class AnyTimeSGD(Optimizer):
    """
    A custom implementation of the AnyTime-SGD optimizer.

    Attributes:
    - gamma (float): Controls the interpolation between parameter values during updates.
    - projection_radius (float or None): Radius for projecting parameter updates to a constrained set.
    - iter (int): Tracks the number of iterations completed.
    - sum_iter (int): Cumulative sum of iteration indices, used in the anytime updates.
    - use_alpha_t (bool): Whether to use the alpha-based weighting scheme for parameter updates.
    - w (list): A deep copy of parameter groups for maintaining intermediate states during updates.
    """

    def __init__(self, params, lr=0.01, gamma=0.9, use_alpha_t=False, weight_decay=0.0,
                 projection_radius=None):
        """
        Initializes the AnyTimeSGD optimizer.

        Args:
        - params (iterable): An iterable of `torch.Tensor`s or `dict`s. Specifies the parameters to optimize.
        - lr (float, optional): Learning rate. Default is 0.01.
        - gamma (float, optional): Interpolation factor between parameter states. Default is 0.9.
        - use_alpha_t (bool, optional): Whether to use the alpha-based interpolation strategy. Default is False.
        - weight_decay (float, optional): L2 regularization coefficient. Default is 0.0.
        - projection_radius (float or None, optional): Radius for projecting parameter updates. Default is None.
        """
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(AnyTimeSGD, self).__init__(params, defaults)
        self.gamma = gamma
        self.projection_radius = projection_radius
        self.iter = 0
        self.sum_iter = 0
        self.use_alpha_t = use_alpha_t
        self.w = []

        # Create a deep copy of parameter groups to maintain intermediate states
        for group in self.param_groups:
            cloned_group = copy.deepcopy(group)
            for i, p in enumerate(group['params']):
                cloned_group['params'][i] = p.clone().detach().requires_grad_(p.requires_grad)

            self.w.append(cloned_group)

    def step(self, closure=None):
        """
        Performs a single optimization step (parameter update).

        Args:
        - closure (callable, optional): A closure that reevaluates the model and returns the loss.
          This is optional and primarily used when loss needs to be recalculated.

        Returns:
        - loss (float or None): The loss value, if the closure is provided; otherwise, None.
        """
        loss = None
        self.iter += 1
        self.sum_iter += self.iter
        if closure is not None:
            loss = closure()

        for group, group_w in zip(self.param_groups, self.w):
            weight_decay = group['weight_decay']
            for p, pw in zip(group['params'], group_w['params']):
                if p.grad is None:
                    continue
                grad = p.grad.data

                # Apply weight decay (L2 regularization)
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                # Get the learning rate for the group
                lr = group['lr']

                # Update intermediate parameters
                pw.data.add_(grad, alpha=-lr)

                # Apply projection if a radius is specified
                if self.projection_radius:
                    pw.data = proj(pw.data, self.projection_radius)

                # Update parameters using alpha-based or gamma-based interpolation
                if self.use_alpha_t:
                    a = self.iter / self.sum_iter
                    b = (self.sum_iter - self.iter) / self.sum_iter
                    p.data = a * pw.data + b * p.data
                else:
                    p.data = self.gamma * pw.data + (1 - self.gamma) * p.data

        return loss
