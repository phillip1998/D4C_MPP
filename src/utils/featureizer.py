import numpy as np
import rdkit.Chem as Chem


class InvalidAtomError(Exception):
    pass

def get_atom_features(mol):
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(atom_feature(atom))
    return np.concatenate(atom_features, axis=0)

def get_bond_features(mol):
    bond_features = []
    for bond in mol.GetBonds():
        bond_features.append(bond_feature(bond))
    return np.concatenate(bond_features, axis=0)


def bond_feature(bond):
    features = np.concatenate([bond_type(bond),
                               bond_stereo(bond)
                            ], axis=0).reshape(1,-1)
    
    return features

def atom_feature(atom):
    features = np.concatenate([atom_symbol_HNums(atom),
                                atom_degree(atom),
                                atom_Aroma(atom),
                                atom_Hybrid(atom),
                                atom_ring(atom),
                                atom_FC(atom)
                            ], axis=0).reshape(1,-1)
    
    return features

def bond_type(bond):
    return np.array(one_of_k_encoding(str(bond.GetBondType()),['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'])).astype(int)

def bond_stereo(bond):
    return np.array(one_of_k_encoding(str(bond.GetStereo()),['STEREONONE', 'STEREOANY', 'STEREOZ', 'STEREOE'])).astype(int)

def atom_symbol_HNums(atom):
    
    return np.array(one_of_k_encoding(atom.GetSymbol(),
                                      ['C', 'N', 'O','S', 'H', 'F', 'Cl', 'Br', 'I','Se','Te','Si','P','B','Sn','Ge'])+
                    one_of_k_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]))


def atom_degree(atom):
    return np.array(one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5 ,6])).astype(int) 


def atom_Aroma(atom):
    return np.array([atom.GetIsAromatic()]).astype(int)


def atom_Hybrid(atom):
    return np.array(one_of_k_encoding(str(atom.GetHybridization()),['S','SP','SP2','SP3','SP3D','SP3D2'])).astype(int)


def atom_ring(atom):
    return np.array([atom.IsInRing()]).astype(int)


def atom_FC(atom):
    return np.array(one_of_k_encoding(atom.GetFormalCharge(), [-4,-3,-2,-1, 0, 1, 2, 3, 4])).astype(int)


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise InvalidAtomError("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))






