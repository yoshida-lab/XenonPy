��
l��F� j�P.�M�.�}q (X   protocol_versionqM�X   little_endianq�X
   type_sizesq}q(X   shortqKX   intqKX   longqKuu.�(X   moduleq ctest_trainer
_Net
qXF   /Users/stephenwu/Documents/GitHub/XenonPy/tests/models/test_trainer.pyqX�  class _Net(torch.nn.Module):

    def __init__(self, n_feature, n_hidden, n_output):
        super().__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)  # output layer

    def forward(self, x_):
        x_ = F.relu(self.hidden(x_))  # activation function for hidden layer
        x_ = self.predict(x_)  # linear output
        return x_
qtqQ)�q}q(X   trainingq�X   _parametersqccollections
OrderedDict
q	)Rq
X   _buffersqh	)RqX   _backward_hooksqh	)RqX   _forward_hooksqh	)RqX   _forward_pre_hooksqh	)RqX   _state_dict_hooksqh	)RqX   _load_state_dict_pre_hooksqh	)RqX   _modulesqh	)Rq(X   hiddenq(h ctorch.nn.modules.linear
Linear
qXU   /opt/miniconda3/envs/XPy37_GPy/lib/python3.7/site-packages/torch/nn/modules/linear.pyqX�	  class Linear(Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
qtqQ)�q}q(h�hh	)Rq (X   weightq!ctorch._utils
_rebuild_parameter
q"ctorch._utils
_rebuild_tensor_v2
q#((X   storageq$ctorch
FloatStorage
q%X   140210914882880q&X   cpuq'K
Ntq(QK K
K�q)KK�q*�h	)Rq+tq,Rq-�h	)Rq.�q/Rq0X   biasq1h"h#((h$h%X   140210914884576q2h'K
Ntq3QK K
�q4K�q5�h	)Rq6tq7Rq8�h	)Rq9�q:Rq;uhh	)Rq<hh	)Rq=hh	)Rq>hh	)Rq?hh	)Rq@hh	)RqAhh	)RqBX   in_featuresqCKX   out_featuresqDK
ubX   predictqEh)�qF}qG(h�hh	)RqH(h!h"h#((h$h%X   140210884024096qIh'K
NtqJQK KK
�qKK
K�qL�h	)RqMtqNRqO�h	)RqP�qQRqRh1h"h#((h$h%X   140210883973200qSh'KNtqTQK K�qUK�qV�h	)RqWtqXRqY�h	)RqZ�q[Rq\uhh	)Rq]hh	)Rq^hh	)Rq_hh	)Rq`hh	)Rqahh	)Rqbhh	)RqchCK
hDKubuub.�]q (X   140210883973200qX   140210884024096qX   140210914882880qX   140210914884576qe.       ��
       l&�=p,�<���:>�O����?�L��W��l��-���
       ��p?T,?�>߾�܀�x�s��>��x�@��wE� �a���>
       ��Ѿ��?8��hIh?�o/?d�W���~���8=g>��r>