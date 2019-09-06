import chainer

from chainer.links import Linear
from chainer_chemistry.config import MAX_ATOMIC_NUM
from chainer_chemistry.links.readout.scatter_ggnn_readout import ScatterGGNNReadout  # NOQA
from chainer_chemistry.links import EmbedAtomID
from chainer_chemistry.links.readout.general_readout import ScatterGeneralReadout  # NOQA


class ReadoutGCNSparse(chainer.Chain):
    def __init__(self, out_dim=64, hidden_channels=None, n_update_layers=None,
                 n_atom_types=MAX_ATOMIC_NUM, n_edge_types=4, input_type='int',
                 scale_adj=False):
        super(ReadoutGCNSparse, self).__init__()
        with self.init_scope():
            if input_type == 'int':
                self.embed = EmbedAtomID(out_size=128,
                                         in_size=n_atom_types)
            elif input_type == 'float':
                self.embed = Linear(None, hidden_channels[0])
            else:
                raise ValueError("[ERROR] Unexpected value input_type={}"
                                 .format(input_type))
            self.rgcn_readout = ScatterGeneralReadout(mode='sum')
            # self.rgcn_readout = ScatterGGNNReadout(
            #     out_dim=out_dim, in_channels=hidden_channels[-1],
            #     nobias=True, activation=functions.tanh)
        # self.num_relations = num_edge_type
        self.input_type = input_type
        self.scale_adj = scale_adj

    def __call__(self, sparse_batch):
        if sparse_batch.x.dtype == self.xp.int32:
            assert self.input_type == 'int'
        else:
            assert self.input_type == 'float'
        h = self.embed(sparse_batch.x)  # (minibatch, max_num_atoms)
        if self.scale_adj:
            raise NotImplementedError
        h = self.rgcn_readout(h, sparse_batch.batch)
        return h
