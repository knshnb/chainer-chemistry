import numpy
import chainer
from chainer_chemistry.dataset.sparse_graph.sparse_graph import SparseGraph


class RelGCNSparseGraph(SparseGraph):
    def __init__(self, x=None, edge_index=None, y=None, edge_attr=None):
        super(RelGCNSparseGraph, self).__init__(x, edge_index, y)
        self.edge_attr = edge_attr

    @classmethod
    def list_to_batch(cls, graph_list):
        batch = super(RelGCNSparseGraph, cls).list_to_batch(graph_list)
        all_edge_attr = [x for graph in graph_list for x in graph.edge_attr]
        batch.edge_attr = numpy.array(all_edge_attr, dtype=numpy.int)
        return batch

    def to_device(self, device):
        super(RelGCNSparseGraph, self).to_device(device)
        self.edge_attr = device.send(self.edge_attr)
        return self

    @staticmethod
    @chainer.dataset.converter()
    def sparse_converter(graph_list, device):
        return RelGCNSparseGraph.sparse_converter_sub(graph_list, device)

    # sparse用のpreprocessorができていないので一時的にRelGCNPreprocessorを流用
    # dense用のdatasetをsparse用(RelGCNSparseGraphのlist)に変換
    @staticmethod
    def dataset_from_dense(dataset):
        graphs = []
        for (x, adj, y) in dataset:
            edge_index = [[], []]
            edge_attr = []
            label_num, n, _ = adj.shape
            for label in range(label_num):
                for i in range(n):
                    for j in range(n):
                        if adj[label, i, j] != 0.:
                            edge_index[0].append(i)
                            edge_index[1].append(j)
                            edge_attr.append(label)
            graphs.append(RelGCNSparseGraph(
                x=x,
                edge_index=numpy.array(edge_index, dtype=numpy.int),
                y=y,
                edge_attr=edge_attr
            ))
        return graphs
