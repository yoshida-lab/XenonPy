#  Copyright (c) 2019. TsumiNa. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.
from torch.optim import lr_scheduler

from xenonpy.model.training.base import BaseLRScheduler

__all__ = ['LambdaLR', 'StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau', 'CyclicLR']


class LambdaLR(BaseLRScheduler):

    def __init__(self, *, lr_lambda, last_epoch=-1):
        """Sets the learning rate of each parameter group to the initial lr
        times a given function. When last_epoch=-1, sets initial lr as lr.

        Args:
            lr_lambda (function or list): A function which computes a multiplicative
                factor given an integer parameter epoch, or a list of such
                functions, one for each group in optimizer.param_groups.
            last_epoch (int): The index of last epoch. Default: -1.

        Example:
            >>> # Assuming optimizer has two groups.
            >>> lambda1 = lambda epoch: epoch // 30
            >>> lambda2 = lambda epoch: 0.95 ** epoch
            >>> scheduler = LambdaLR(lr_lambda=[lambda1, lambda2])
            >>> scheduler(optimizer)
            >>> for epoch in range(100):
            >>>     train(...)
            >>>     validate(...)
            >>>     scheduler.step(),
        """
        super().__init__(lr_scheduler.LambdaLR, lr_lambda=lr_lambda, last_epoch=last_epoch)


class StepLR(BaseLRScheduler):

    def __init__(self, *, step_size, gamma=0.1, last_epoch=-1):
        """Decays the learning rate of each parameter group by gamma every
        step_size epochs. Notice that such decay can happen simultaneously with
        other changes to the learning rate from outside this scheduler. When
        last_epoch=-1, sets initial lr as lr.

        Args:
            step_size (int): Period of learning rate decay.
            gamma (float): Multiplicative factor of learning rate decay.
                Default: 0.1.
            last_epoch (int): The index of last epoch. Default: -1.

        Example:
            >>> # Assuming optimizer uses lr = 0.05 for all groups
            >>> # lr = 0.05     if epoch < 30
            >>> # lr = 0.005    if 30 <= epoch < 60
            >>> # lr = 0.0005   if 60 <= epoch < 90
            >>> # ...
            >>> scheduler = StepLR(step_size=30, gamma=0.1)
            >>> scheduler(optimizer)
            >>> for epoch in range(100):
            >>>     train(...)
            >>>     validate(...)
            >>>     scheduler.step()
        """
        super().__init__(lr_scheduler.StepLR, step_size=step_size, gamma=gamma, last_epoch=last_epoch)


class MultiStepLR(BaseLRScheduler):

    def __init__(self, *, milestones, gamma=0.1, last_epoch=-1):
        """Decays the learning rate of each parameter group by gamma once the
        number of epoch reaches one of the milestones. Notice that such decay can
        happen simultaneously with other changes to the learning rate from outside
        this scheduler. When last_epoch=-1, sets initial lr as lr.

        Args:
            milestones (list): List of epoch indices. Must be increasing.
            gamma (float): Multiplicative factor of learning rate decay.
                Default: 0.1.
            last_epoch (int): The index of last epoch. Default: -1.

        Example:
            >>> # Assuming optimizer uses lr = 0.05 for all groups
            >>> # lr = 0.05     if epoch < 30
            >>> # lr = 0.005    if 30 <= epoch < 80
            >>> # lr = 0.0005   if epoch >= 80
            >>> scheduler = MultiStepLR(milestones=[30,80], gamma=0.1)
            >>> scheduler(optimizer)
            >>> for epoch in range(100):
            >>>     train(...)
            >>>     validate(...)
            >>>     scheduler.step(),
        """
        super().__init__(lr_scheduler.MultiStepLR, milestones=milestones, gamma=gamma, last_epoch=last_epoch)


class ExponentialLR(BaseLRScheduler):

    def __init__(self, *, gamma, last_epoch=-1):
        """Decays the learning rate of each parameter group by gamma every epoch.
        When last_epoch=-1, sets initial lr as lr.

        Args:
            gamma (float): Multiplicative factor of learning rate decay.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        super().__init__(lr_scheduler.ExponentialLR, gamma=gamma, last_epoch=last_epoch)


class CosineAnnealingLR(BaseLRScheduler):

    def __init__(self, *, T_max, eta_min=0, last_epoch=-1):
        r"""Set the learning rate of each parameter group using a cosine annealing
        schedule, where :math:`\eta_{max}` is set to the initial lr and
        :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

        .. math::
            \eta_{t+1} = \eta_{min} + (\eta_t - \eta_{min})\frac{1 +
            \cos(\frac{T_{cur+1}}{T_{max}}\pi)}{1 + \cos(\frac{T_{cur}}{T_{max}}\pi)},
            T_{cur} \neq (2k+1)T_{max};\\
            \eta_{t+1} = \eta_{t} + (\eta_{max} - \eta_{min})\frac{1 -
            \cos(\frac{1}{T_{max}}\pi)}{2},
            T_{cur} = (2k+1)T_{max}.\\

        When last_epoch=-1, sets initial lr as lr. Notice that because the schedule
        is defined recursively, the learning rate can be simultaneously modified
        outside this scheduler by other operators. If the learning rate is set
        solely by this scheduler, the learning rate at each step becomes:

        .. math::
            \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
            \cos(\frac{T_{cur}}{T_{max}}\pi))

        It has been proposed in
        `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
        implements the cosine annealing part of SGDR, and not the restarts.

        Args:
           T_max (int): Maximum number of iterations.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.

        .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
            https://arxiv.org/abs/1608.03983
        """
        super().__init__(lr_scheduler.CosineAnnealingLR, T_max=T_max, eta_min=eta_min, last_epoch=last_epoch)


class ReduceLROnPlateau(BaseLRScheduler):

    def __init__(self, *, mode='min', factor=0.1, patience=10,
                 verbose=False, threshold=1e-4, threshold_mode='rel',
                 cooldown=0, min_lr=0, eps=1e-8):
        """Reduce learning rate when a metric has stopped improving.
        Models often benefit from reducing the learning rate by a factor
        of 2-10 once learning stagnates. This scheduler reads a metrics
        quantity and if no improvement is seen for a 'patience' number
        of epochs, the learning rate is reduced.

        Args:
            mode (str): One of `min`, `max`. In `min` mode, lr will
                be reduced when the quantity monitored has stopped
                decreasing; in `max` mode it will be reduced when the
                quantity monitored has stopped increasing. Default: 'min'.
            factor (float): Factor by which the learning rate will be
                reduced. new_lr = lr * factor. Default: 0.1.
            patience (int): Number of epochs with no improvement after
                which learning rate will be reduced. For example, if
                `patience = 2`, then we will ignore the first 2 epochs
                with no improvement, and will only decrease the LR after the
                3rd epoch if the loss still hasn't improved then.
                Default: 10.
            verbose (bool): If ``True``, prints a message to stdout for
                each update. Default: ``False``.
            threshold (float): Threshold for measuring the new optimum,
                to only focus on significant changes. Default: 1e-4.
            threshold_mode (str): One of `rel`, `abs`. In `rel` mode,
                dynamic_threshold = best * ( 1 + threshold ) in 'max'
                mode or best * ( 1 - threshold ) in `min` mode.
                In `abs` mode, dynamic_threshold = best + threshold in
                `max` mode or best - threshold in `min` mode. Default: 'rel'.
            cooldown (int): Number of epochs to wait before resuming
                normal operation after lr has been reduced. Default: 0.
            min_lr (float or list): A scalar or a list of scalars. A
                lower bound on the learning rate of all param groups
                or each group respectively. Default: 0.
            eps (float): Minimal decay applied to lr. If the difference
                between new and old lr is smaller than eps, the update is
                ignored. Default: 1e-8.

        """
        super().__init__(lr_scheduler.ReduceLROnPlateau, mode=mode, factor=factor, patience=patience,
                         verbose=verbose, threshold=threshold, threshold_mode=threshold_mode,
                         cooldown=cooldown, min_lr=min_lr, eps=eps)


class CyclicLR(BaseLRScheduler):

    def __init__(self, *, base_lr,
                 max_lr,
                 step_size_up=2000,
                 step_size_down=None,
                 mode='triangular',
                 gamma=1.,
                 scale_fn=None,
                 scale_mode='cycle',
                 cycle_momentum=True,
                 base_momentum=0.8,
                 max_momentum=0.9,
                 last_epoch=-1):
        """Sets the learning rate of each parameter group according to
        cyclical learning rate policy (CLR). The policy cycles the learning
        rate between two boundaries with a constant frequency, as detailed in
        the paper `Cyclical Learning Rates for Training Neural Networks`_.
        The distance between the two boundaries can be scaled on a per-iteration
        or per-cycle basis.

        Cyclical learning rate policy changes the learning rate after every batch.
        `step` should be called after a batch has been used for training.

        This class has three built-in policies, as put forth in the paper:

        "triangular":
            A basic triangular cycle w/ no amplitude scaling.
        "triangular2":
            A basic triangular cycle that scales initial amplitude by half each cycle.
        "exp_range":
            A cycle that scales initial amplitude by gamma**(cycle iterations) at each
            cycle iteration.

        This implementation was adapted from the github repo: `bckenstler/CLR`_

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            base_lr (float or list): Initial learning rate which is the
                lower boundary in the cycle for each parameter group.
            max_lr (float or list): Upper learning rate boundaries in the cycle
                for each parameter group. Functionally,
                it defines the cycle amplitude (max_lr - base_lr).
                The lr at any cycle is the sum of base_lr
                and some scaling of the amplitude; therefore
                max_lr may not actually be reached depending on
                scaling function.
            step_size_up (int): Number of training iterations in the
                increasing half of a cycle. Default: 2000
            step_size_down (int): Number of training iterations in the
                decreasing half of a cycle. If step_size_down is None,
                it is set to step_size_up. Default: None
            mode (str): One of {triangular, triangular2, exp_range}.
                Values correspond to policies detailed above.
                If scale_fn is not None, this argument is ignored.
                Default: 'triangular'
            gamma (float): Constant in 'exp_range' scaling function:
                gamma**(cycle iterations)
                Default: 1.0
            scale_fn (function): Custom scaling policy defined by a single
                argument lambda function, where
                0 <= scale_fn(x) <= 1 for all x >= 0.
                If specified, then 'mode' is ignored.
                Default: None
            scale_mode (str): {'cycle', 'iterations'}.
                Defines whether scale_fn is evaluated on
                cycle number or cycle iterations (training
                iterations since start of cycle).
                Default: 'cycle'
            cycle_momentum (bool): If ``True``, momentum is cycled inversely
                to learning rate between 'base_momentum' and 'max_momentum'.
                Default: True
            base_momentum (float or list): Initial momentum which is the
                lower boundary in the cycle for each parameter group.
                Default: 0.8
            max_momentum (float or list): Upper momentum boundaries in the cycle
                for each parameter group. Functionally,
                it defines the cycle amplitude (max_momentum - base_momentum).
                The momentum at any cycle is the difference of max_momentum
                and some scaling of the amplitude; therefore
                base_momentum may not actually be reached depending on
                scaling function. Default: 0.9
            last_epoch (int): The index of the last batch. This parameter is used when
                resuming a training job. Since `step()` should be invoked after each
                batch instead of after each epoch, this number represents the total
                number of *batches* computed, not the total number of epochs computed.
                When last_epoch=-1, the schedule is started from the beginning.
                Default: -1

        .. _Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
        .. _bckenstler/CLR: https://github.com/bckenstler/CLR
        """
        super().__init__(lr_scheduler.CyclicLR, base_lr=base_lr,
                         max_lr=max_lr,
                         step_size_up=step_size_up,
                         step_size_down=step_size_down,
                         mode=mode,
                         gamma=gamma,
                         scale_fn=scale_fn,
                         scale_mode=scale_mode,
                         cycle_momentum=cycle_momentum,
                         base_momentum=base_momentum,
                         max_momentum=max_momentum,
                         last_epoch=last_epoch)
