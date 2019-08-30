import chainer
from chainer import functions
from chainer_chemistry.links.readout.scatter_ggnn_readout import ScatterGGNNReadout
from chainer_chemistry.links.connection.embed_atom_id import EmbedAtomID
from chainer_chemistry.links.layers.simple_relgcn_sparse_layer import SimpleRelGCNSparseLayer


class SimpleRelGCNSparse(chainer.Chain):
    """
    simple RelGCN without edge label and degree normalization
    """

    def __init__(self, out_dim, hidden_channels=[16, 128, 64, 1]):
        super(SimpleRelGCNSparse, self).__init__()
        with self.init_scope():
            self.embed = EmbedAtomID(out_size=hidden_channels[0])
            self.rgcn_convs = chainer.ChainList(*[
                SimpleRelGCNLayer(hidden_channels[i], hidden_channels[i + 1])
                for i in range(len(hidden_channels) - 1)])
            self.rgcn_readout = ScatterGGNNReadout(
                out_dim=out_dim, in_channels=hidden_channels[-1],
                nobias=True, activation=functions.tanh)

    def __call__(self, graph):
        h = self.embed(graph.x)
        for i, rgcn_conv in enumerate(self.rgcn_convs):
            h = rgcn_conv(h, graph)
            if i != len(self.rgcn_convs) - 1:
                h = chainer.functions.relu(h)
        return self.rgcn_readout(h, graph.batch)
