import numpy as np
import chainer
from chainer import functions


class ScatterGeneralReadout(chainer.Link):
    """Implementation of readout by scatter operation
    """

    def __init__(self, mode='sum', activation=None, **kwargs):
        super(ScatterGeneralReadout, self).__init__()
        self.mode = mode
        self.activation = activation

    def __call__(self, h, batch, **kwargs):
        if self.activation is not None:
            h = self.activation(h)
        else:
            h = h

        if self.mode == 'sum':
            y = np.zeros((batch[-1] + 1, h.shape[1]), dtype=np.float32)
            y = functions.scatter_add(y, batch, h)
        elif self.mode == 'max':
            raise NotImplementedError
        elif self.mode == 'summax':
            raise NotImplementedError
        else:
            raise ValueError('mode {} is not supported'.format(self.mode))
        return y
