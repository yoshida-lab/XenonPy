# Copyright 2017 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# %%
from pymonad.Maybe import Nothing, Maybe, _Nothing
from pymonad.Monad import Monad


class Just(Maybe):
    """ The 'Maybe' type used to represent a calculation that has succeeded. """

    def __init__(self, *args, **kwargs):
        """
        Creates a Just value representing a successful calculation.
        'value' can be any type of value, including functions.
        """
        super(Maybe, self).__init__((args, kwargs))

    def __str__(self):
        return "Just " + str(self.getValue())

    def __eq__(self, other):
        super(Just, self).__eq__(other)
        if isinstance(other, _Nothing):
            return False
        elif self.getValue() == other.getValue():
            return True
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __invert__(self):
        """
        '~' let return out of monad
        """
        return self.getValue()

    def __lshift__(self, value):
        """
        The 'opposite bind' operator. The following are equivalent:
            someFunction << monadValue
            someFunction(monadValue.getValue)
        """

        if callable(value) or not isinstance(value, Monad):
            raise TypeError(
                "right side of Operator '<<' must be a Monad Value.")
        if not callable(self.getValue()):
            raise TypeError(
                "left side of Operator '<<' must be a function return Monad value."
            )
        return self.getValue()(*value.value[0], **value.value[1])

    def fmap(self, function):
        """ Applies 'function' to the 'Just' value and returns a new 'Just' value. """

        return Just(function(self.getValue()))

    def amap(self, functorValue):
        """
        Applies the function stored in the functor to the value of 'functorValue',
        returning a new 'Just' value.
        """

        return self.getValue() * functorValue

    def bind(self, function):
        """ Applies 'function' to a 'Just' value.
        'function' must accept a single argument and return a 'Maybe' type,
        either 'Just(something)' or 'Nothing'.
        """

        return function(*self.value[0], **self.value[1])

    def getValue(self):
        ret = [arg for arg in self.value[0]]
        if self.value[1]:
            ret.append(self.value[1])
        if len(ret) == 1:
            return ret[0]
        return tuple(ret)

    def mplus(self, other):
        """
        Combines Maybe monoid values into a single monoid value.
        The Maybe monoid works when the values it contains are also monoids with a defined mzero and mplus.
        This allows you do things like::

            >>> Just(1) + Just(9) == Just(10)
            >>> Just("Hello ") + Just("World") == Just("Hello World")
            >>> Just([1, 2, 3]) + Just([4, 5, 6]) == Just([1, 2, 3, 4, 5, 6])

        The identity value is :class:`Nothing`::

            >>> Just(1) + Nothing == Just(1)
        """

        if other == Nothing:
            return self
        else:
            return Just(self.value + other.value)


def combinator(cls_func):
    """
    wrap model class or functions to monad
    """

    def __truediv__(self, func):
        """
        The 'pass' operator. The following are equivalent:
            someFunction / someFunction
        """
        if callable(func):

            def _(*args, **kwargs):
                _ = self(*args, **kwargs)
                return func(*args, **kwargs)

            return _
        else:
            raise TypeError("left side of Operator '/' must be a function.")

    def __invert__(self):
        """
        '~' let return out of monad
        """
        self.inv = True
        return self

    def _cls_deco(cls):
        cls_name = cls.__name__

        def __call__(self, *args, **kwargs):
            if not self.trained:
                if hasattr(self, 'fit_transform'):
                    self.trained = True
                    ret = self.fit_transform(*args, **kwargs)
                    return Just(ret) if not self.inv else ret
                if hasattr(self, 'fit'):
                    self.trained = True
                    self.fit(*args, **kwargs)
                    return Just(self) if not self.inv else self
                raise ValueError(
                    'combinator class must implement fit-transform/predict')
            if hasattr(self, 'predict'):
                ret = self.predict(*args, **kwargs)
                return Just(ret) if not self.inv else ret
            if hasattr(self, 'transform'):
                ret = self.transform(*args, **kwargs)
                return Just(ret) if not self.inv else ret
            if hasattr(self, 'draw'):
                ret = self.draw(*args, **kwargs)
                return Just(ret) if not self.inv else ret
            raise ValueError(
                'combinator class must implement fit-transform/predict')

        return type(cls_name, (cls,),
                    dict(
                        trained=False,
                        inv=False,
                        __init__=cls.__init__,
                        __call__=__call__,
                        __invert__=__invert__,
                        __truediv__=__truediv__, ))

    def _func_deco(func):
        func_name = func.__name__

        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def __call__(self, *args, **kwargs):
            args_ = args + self.args
            kwargs_ = dict(kwargs, **self.kwargs)
            return Just(func(*args_, **kwargs_))

        return type(func_name, (object,),
                    dict(
                        inv=False,
                        __init__=__init__,
                        __call__=__call__,
                        __invert__=__invert__,
                        __truediv__=__truediv__))

    if isinstance(cls_func, type):
        return _cls_deco(cls_func)
    from types import FunctionType
    if isinstance(cls_func, FunctionType):
        return _func_deco(cls_func)
