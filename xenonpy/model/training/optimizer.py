#  Copyright (c) 2019. TsumiNa. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from torch import optim

from xenonpy.model.training.base import BaseOptimizer

__all__ = ['Adadelta', 'Adagrad', 'Adam', 'Adamax', 'ASGD', 'SGD', 'SparseAdam', 'RMSprop', 'Rprop', 'LBFGS']


class Adadelta(BaseOptimizer):

    def __init__(self, *, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0):
        """Implements Adadelta algorithm.

        It has been proposed in `ADADELTA: An Adaptive Learning Rate Method`__.

        Arguments:
            rho (float, optional): coefficient used for computing a running average
                of squared gradients (default: 0.9)
            eps (float, optional): term added to the denominator to improve
                numerical stability (default: 1e-6)
            lr (float, optional): coefficient that scale delta before it is applied
                to the parameters (default: 1.0)
            weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

        __ https://arxiv.org/abs/1212.5701
        """
        super().__init__(optim.Adadelta, lr=lr, rho=rho, eps=eps, weight_decay=weight_decay)


class Adagrad(BaseOptimizer):

    def __init__(self, *, lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0):
        """Implements Adagrad algorithm.

        It has been proposed in `Adaptive Subgradient Methods for Online Learning
        and Stochastic Optimization`_.

        Arguments:
            lr (float, optional): learning rate (default: 1e-2)
            lr_decay (float, optional): learning rate decay (default: 0)
            weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

        .. _Adaptive Subgradient Methods for Online Learning and Stochastic
            Optimization: http://jmlr.org/papers/v12/duchi11a.html
        """
        super().__init__(optim.Adagrad, lr=lr, lr_decay=lr_decay, weight_decay=weight_decay,
                         initial_accumulator_value=initial_accumulator_value)


class Adam(BaseOptimizer):

    def __init__(self, *, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        r"""Implements Adam algorithm.

        It has been proposed in `Adam: A Method for Stochastic Optimization`_.

        Arguments:
            lr (float, optional): learning rate (default: 1e-3)
            betas (Tuple[float, float], optional): coefficients used for computing
                running averages of gradient and its square (default: (0.9, 0.999))
            eps (float, optional): term added to the denominator to improve
                numerical stability (default: 1e-8)
            weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
            amsgrad (boolean, optional): whether to use the AMSGrad variant of this
                algorithm from the paper `On the Convergence of Adam and Beyond`_
                (default: False)

        .. _Adam\: A Method for Stochastic Optimization:
            https://arxiv.org/abs/1412.6980
        .. _On the Convergence of Adam and Beyond:
            https://openreview.net/forum?id=ryQu7f-RZ
        """

        super().__init__(optim.Adam, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)


class SparseAdam(BaseOptimizer):

    def __init__(self, *, lr=0.001, betas=(0.9, 0.999), eps=1e-08):
        r"""Implements lazy version of Adam algorithm suitable for sparse tensors.

        In this variant, only moments that show up in the gradient get updated, and
        only those portions of the gradient get applied to the parameters.

        Arguments:
            lr (float, optional): learning rate (default: 1e-3)
            betas (Tuple[float, float], optional): coefficients used for computing
                running averages of gradient and its square (default: (0.9, 0.999))
            eps (float, optional): term added to the denominator to improve
                numerical stability (default: 1e-8)

        .. _Adam\: A Method for Stochastic Optimization:
            https://arxiv.org/abs/1412.6980
        """

        super().__init__(optim.SparseAdam, lr=lr, betas=betas, eps=eps)


class Adamax(BaseOptimizer):

    def __init__(self, *, lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0):
        """Implements Adamax algorithm (a variant of Adam based on infinity norm).

        It has been proposed in `Adam: A Method for Stochastic Optimization`__.

        Arguments:
            lr (float, optional): learning rate (default: 2e-3)
            betas (Tuple[float, float], optional): coefficients used for computing
                running averages of gradient and its square
            eps (float, optional): term added to the denominator to improve
                numerical stability (default: 1e-8)
            weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

        __ https://arxiv.org/abs/1412.6980
        """

        super().__init__(optim.Adamax, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


class ASGD(BaseOptimizer):

    def __init__(self, *, lr=0.002, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0):
        """Implements Averaged Stochastic Gradient Descent.

        It has been proposed in `Acceleration of stochastic approximation by
        averaging`_.

        Arguments:
            lr (float, optional): learning rate (default: 1e-2)
            lambd (float, optional): decay term (default: 1e-4)
            alpha (float, optional): power for eta update (default: 0.75)
            t0 (float, optional): point at which to start averaging (default: 1e6)
            weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

        .. _Acceleration of stochastic approximation by averaging:
            http://dl.acm.org/citation.cfm?id=131098
        """

        super().__init__(optim.ASGD, lr=lr, lambd=lambd, alpha=alpha, t0=t0, weight_decay=weight_decay)


class LBFGS(BaseOptimizer):

    def __init__(self, *, lr=1, max_iter=20, max_eval=None,
                 tolerance_grad=1e-5, tolerance_change=1e-9, history_size=100,
                 line_search_fn=None):
        """Implements L-BFGS algorithm.

        .. warning::
            This optimizer doesn't support per-parameter options and parameter
            groups (there can be only one).

        .. warning::
            Right now all parameters have to be on a single device. This will be
            improved in the future.

        .. note::
            This is a very memory intensive optimizer (it requires additional
            ``param_bytes * (history_size + 1)`` bytes). If it doesn't fit in memory
            try reducing the history size, or use a different algorithm.

        Arguments:
            lr (float): learning rate (default: 1)
            max_iter (int): maximal number of iterations per optimization step
                (default: 20)
            max_eval (int): maximal number of function evaluations per optimization
                step (default: max_iter * 1.25).
            tolerance_grad (float): termination tolerance on first order optimality
                (default: 1e-5).
            tolerance_change (float): termination tolerance on function
                value/parameter changes (default: 1e-9).
            history_size (int): update history size (default: 100).
        """

        super().__init__(optim.LBFGS, lr=lr, max_iter=max_iter, max_eval=max_eval,
                         tolerance_grad=tolerance_grad, tolerance_change=tolerance_change, history_size=history_size,
                         line_search_fn=line_search_fn)


class RMSprop(BaseOptimizer):

    def __init__(self, *, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False):
        """Implements RMSprop algorithm.

        Proposed by G. Hinton in his
        `course <http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_.

        The centered version first appears in `Generating Sequences
        With Recurrent Neural Networks <https://arxiv.org/pdf/1308.0850v5.pdf>`_.

        Arguments:
            lr (float, optional): learning rate (default: 1e-2)
            momentum (float, optional): momentum factor (default: 0)
            alpha (float, optional): smoothing constant (default: 0.99)
            eps (float, optional): term added to the denominator to improve
                numerical stability (default: 1e-8)
            centered (bool, optional) : if ``True``, compute the centered RMSProp,
                the gradient is normalized by an estimation of its variance
            weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

        """

        super().__init__(optim.RMSprop, lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum,
                         centered=centered)


class Rprop(BaseOptimizer):

    def __init__(self, *, lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50)):
        """Implements the resilient backpropagation algorithm.

        Arguments:
            lr (float, optional): learning rate (default: 1e-2)
            etas (Tuple[float, float], optional): pair of (etaminus, etaplis), that
                are multiplicative increase and decrease factors
                (default: (0.5, 1.2))
            step_sizes (Tuple[float, float], optional): a pair of minimal and
                maximal allowed step sizes (default: (1e-6, 50))
        """

        super().__init__(optim.Rprop, lr=lr, etas=etas, step_sizes=step_sizes)


class SGD(BaseOptimizer):

    def __init__(self, *, lr=0.01, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        r"""Implements stochastic gradient descent (optionally with momentum).

        Nesterov momentum is based on the formula from
        `On the importance of initialization and momentum in deep learning`__.

        Args:
            lr (float): learning rate
            momentum (float, optional): momentum factor (default: 0)
            weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
            dampening (float, optional): dampening for momentum (default: 0)
            nesterov (bool, optional): enables Nesterov momentum (default: False)

        Example:
            >>> optimizer = torch.optim.SGD(lr=0.1, momentum=0.9)
            >>> optimizer(model.parameters())
            >>> optimizer.zero_grad()
            >>> loss_fn(model(input), target).backward()
            >>> optimizer.step(),

        __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

        .. note::
            The implementation of SGD with Momentum/Nesterov subtly differs from
            Sutskever et. al. and implementations in some other frameworks.

            Considering the specific case of Momentum, the update can be written as

            .. math::
                      v = \rho * v + g \\
                      p = p - lr * v

            where p, g, v and :math:`\rho` denote the parameters, gradient,
            velocity, and momentum respectively.

            This is in contrast to Sutskever et. al. and
            other frameworks which employ an update of the form

            .. math::
                 v = \rho * v + lr * g \\
                 p = p - v

            The Nesterov version is analogously modified.
        """

        super().__init__(optim.SGD, lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay,
                         nesterov=nesterov)
