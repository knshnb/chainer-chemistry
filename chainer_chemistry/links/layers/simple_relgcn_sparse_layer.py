import chainer


class SimpleRelGCNSparseLayer(chainer.Chain):
    def __init__(self, in_channels, out_channels):
        super(SimpleRelGCNSparseLayer, self).__init__()
        with self.init_scope():
            self.root_weight = chainer.links.Linear(in_channels, out_channels)
            self.edge_weight = chainer.links.Linear(in_channels, out_channels)

    def __call__(self, h, graph):
        next_h = self.root_weight(h)
        messages = self.edge_weight(h)[graph.edge_index[0]]
        return chainer.functions.scatter_add(next_h, graph.edge_index[1],
                                             messages)
