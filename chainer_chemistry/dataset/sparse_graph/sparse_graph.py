import numpy as np
import chainer


class SparseGraph(object):
    def __init__(self, x=None, edge_index=None, y=None):
        self.x = x
        self.edge_index = edge_index
        self.y = y

    @property
    def num_nodes(self):
        return self.x.shape[0]

    @property
    def num_edges(self):
        return self.edge_index.shape[1]

    def __repr__(self):
        info = ['{}={}'.format(key, item)
                for key, item in self.__dict__.items()]
        return '{}({})'.format(self.__class__.__name__, ', '.join(info))

    @classmethod
    def list_to_batch(cls, graph_list):
        batch = cls()
        num_all_nodes = sum([graph.num_nodes for graph in graph_list])
        num_all_edges = sum([graph.num_edges for graph in graph_list])
        batch.x = np.empty((num_all_nodes,), dtype=np.int32)
        batch.edge_index = np.empty((2, num_all_edges), dtype=np.int)
        if graph_list[0].y is not None:
            batch.y = np.vectorize(lambda graph: graph.y)(graph_list).reshape(
                (len(graph_list), graph_list[0].y.shape[0]))
        batch.batch = np.empty((num_all_nodes,), dtype=np.int)

        cur_node = 0
        cur_edge = 0
        for i, graph in enumerate(graph_list):
            num_nodes = graph.num_nodes
            num_edges = graph.num_edges
            batch.x[cur_node: cur_node + num_nodes] = graph.x
            batch.edge_index[:, cur_edge: cur_edge + num_edges
                             ] = graph.edge_index + cur_node
            batch.batch[cur_node: cur_node + num_nodes] = i
            cur_node += num_nodes
            cur_edge += num_edges
        return batch

    def to_device(self, device):
        device.send(self.x)
        device.send(self.edge_index)
        device.send(self.y)
        device.send(self.batch)
        return self

    @classmethod
    def sparse_converter_sub(cls, graph_list, device):
        labels = np.array([graph.y for graph in graph_list], dtype=np.float32)
        device.send(labels)
        return cls.list_to_batch(graph_list).to_device(device), labels

    @staticmethod
    @chainer.dataset.converter()
    def sparse_converter(graph_list, device):
        return SparseGraph.sparse_converter_sub(graph_list, device)
