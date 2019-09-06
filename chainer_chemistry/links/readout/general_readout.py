import chainer
from chainer import functions, links


class GeneralReadout(chainer.Link):
    """General submodule for readout part.

    This class can be used for `rsgcn` and `weavenet`.
    Note that this class has no learnable parameter,
    even though this is subclass of `chainer.Link`.
    This class is under `links` module for consistency
    with other readout module.

    Args:
        mode (str):
        activation (callable): activation function
    """

    def __init__(self, mode='sum', activation=None, **kwargs):
        super(GeneralReadout, self).__init__()
        self.mode = mode
        self.activation = activation

    def __call__(self, h, axis=1, **kwargs):
        if self.activation is not None:
            h = self.activation(h)
        else:
            h = h

        if self.mode == 'sum':
            y = functions.sum(h, axis=axis)
        elif self.mode == 'max':
            y = functions.max(h, axis=axis)
        elif self.mode == 'summax':
            h_sum = functions.sum(h, axis=axis)
            h_max = functions.max(h, axis=axis)
            y = functions.concat((h_sum, h_max), axis=axis)
        else:
            raise ValueError('mode {} is not supported'.format(self.mode))
        return y


class ScatterGeneralReadout(chainer.Link):
    def __init__(self, mode='sum', activation=None, concat_n_info=False,
                 add_n_info=False, **kwargs):
        self.concat_n_info = concat_n_info
        self.add_n_info = add_n_info
        super(ScatterGeneralReadout, self).__init__()
        with self.init_scope():
            if self.add_n_info:
                self.linear1 = links.Linear(1, 1)
                self.linear2 = links.Linear(1, 128)
        self.mode = mode
        self.activation = activation

    def __call__(self, h, batch, axis=1, **kwargs):
        if self.activation is not None:
            h = self.activation(h)
        else:
            h = h

        if self.mode == 'sum':
            y = self.xp.zeros((batch[-1] + 1, h.shape[1]),
                              dtype=self.xp.float32)
            y = functions.scatter_add(y, batch, h)
        else:
            raise ValueError('mode {} is not supported'.format(self.mode))

        if self.concat_n_info:
            n_nodes = self.xp.zeros(y.shape[0], dtype=self.xp.float32)
            n_nodes = functions.scatter_add(n_nodes, batch,
                                            self.xp.ones(batch.shape[0]))
            y = functions.concat((y, n_nodes.reshape((-1, 1))))

        if self.add_n_info:
            n_nodes = self.xp.zeros(y.shape[0], dtype=self.xp.float32)
            n_nodes = functions.scatter_add(n_nodes, batch,
                                            self.xp.ones(batch.shape[0]))
            n_nodes = n_nodes.reshape((-1, 1))
            y = y + self.linear2(9 - n_nodes)
            # y = y + self.linear2(-n_nodes)
            # y = y + self.linear2(self.linear1(n_nodes))
        return y
