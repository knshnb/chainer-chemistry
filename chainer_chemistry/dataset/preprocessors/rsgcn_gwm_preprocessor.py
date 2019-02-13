from chainer_chemistry.dataset.preprocessors.common import construct_adj_matrix  # NOQA
from chainer_chemistry.dataset.preprocessors.common import construct_atomic_number_array  # NOQA
from chainer_chemistry.dataset.preprocessors.common import type_check_num_atoms  # NOQA
from chainer_chemistry.dataset.preprocessors.common import construct_supernode_feature # NOQA
from chainer_chemistry.dataset.preprocessors.mol_preprocessor import MolPreprocessor  # NOQA


import numpy


class RSGCNGWMPreprocessor(MolPreprocessor):
    """RSGCN+GWM Preprocessor

    Args:
        max_atoms (int): Max number of atoms for each molecule, if the
            number of atoms is more than this value, this data is simply
            ignored.
            Setting negative value indicates no limit for max atoms.
        out_size (int): It specifies the size of array returned by
            `get_input_features`.
            If the number of atoms in the molecule is less than this value,
            the returned arrays is padded to have fixed size.
            Setting negative value indicates do not pad returned array.
        add_Hs (bool): If True, implicit Hs are added.
        out_size_super (int): indicate the length of the super node feature.
        kekulize (bool): If True, Kekulizes the molecule.

    """

    def __init__(self, max_atoms=-1, out_size=-1, out_size_super=-1, add_Hs=False,
                 kekulize=False):
        super(RSGCNGWMPreprocessor, self).__init__(
            add_Hs=add_Hs, kekulize=kekulize)
        if max_atoms >= 0 and out_size >= 0 and max_atoms > out_size:
            raise ValueError('max_atoms {} must be less or equal to '
                             'out_size {}'.format(max_atoms, out_size))
        self.max_atoms = max_atoms
        self.out_size = out_size
        self.out_size_super = out_size_super

    def get_input_features(self, mol):
        """get input features

        Args:
            mol (Mol):

        Returns:

        """
        type_check_num_atoms(mol, self.max_atoms)
        num_atoms = mol.GetNumAtoms()

        # Construct the atom array and adjacency matrix.
        atom_array = construct_atomic_number_array(mol, out_size=self.out_size)
        adj_array = construct_adj_matrix(mol, out_size=self.out_size)

        # Adjust the adjacency matrix.
        degree_vec = numpy.sum(adj_array[:num_atoms], axis=1)
        degree_sqrt_inv = 1. / numpy.sqrt(degree_vec)

        adj_array[:num_atoms, :num_atoms] *= numpy.broadcast_to(
            degree_sqrt_inv[:, None], (num_atoms, num_atoms))
        adj_array[:num_atoms, :num_atoms] *= numpy.broadcast_to(
            degree_sqrt_inv[None, :], (num_atoms, num_atoms))
        super_node_x = construct_supernode_feature(mol, atom_array, adj_array, out_size=self.out_size_super)

        return atom_array, adj_array, super_node_x
