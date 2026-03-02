import os.path as osp
import pickle
from typing import Any, Sequence

# NOTE: RDKit is only required for molecular datasets. Import lazily so that
# non-molecular datasets (e.g., ACM_subgraphs) can run even if RDKit is missing
# or has binary compatibility issues on the system.
try:
    from rdkit import Chem  # type: ignore
except Exception as _e:  # pragma: no cover
    Chem = None  # type: ignore
    _RDKIT_IMPORT_ERROR = _e  # noqa: N816
import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph


def mol_to_torch_geometric(mol, atom_encoder, smiles):
    if Chem is None:  # pragma: no cover
        raise ImportError( 
            "RDKit is required for molecular datasets, but it could not be imported. "
            f"Original error: {_RDKIT_IMPORT_ERROR}"
        )
    adj = torch.from_numpy(Chem.rdmolops.GetAdjacencyMatrix(mol, useBO=True))
    edge_index = adj.nonzero().contiguous().T
    bond_types = adj[edge_index[0], edge_index[1]]
    bond_types[bond_types == 1.5] = 4
    edge_attr = bond_types.long()

    node_types = []
    all_charge = []
    for atom in mol.GetAtoms():
        node_types.append(atom_encoder[atom.GetSymbol()])
        all_charge.append(atom.GetFormalCharge())

    node_types = torch.Tensor(node_types).long()
    all_charge = torch.Tensor(all_charge).long()

    data = Data(
        x=node_types,
        edge_index=edge_index,
        edge_attr=edge_attr,
        charge=all_charge,
        smiles=smiles,
    )
    return data


def remove_hydrogens(data: Data):
    to_keep = data.x > 0
    new_edge_index, new_edge_attr = subgraph(
        to_keep,
        data.edge_index,
        data.edge_attr,
        relabel_nodes=True,
        num_nodes=len(to_keep),
    )
    return Data(
        x=data.x[to_keep] - 1,  # Shift onehot encoding to match atom decoder
        charge=data.charge[to_keep],
        edge_index=new_edge_index,
        edge_attr=new_edge_attr,
    )


def save_pickle(array, path):
    with open(path, "wb") as f:
        pickle.dump(array, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def files_exist(files) -> bool:
    return len(files) != 0 and all([osp.exists(f) for f in files])


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


class Statistics:
    def __init__(
        self, num_nodes, node_types, bond_types, charge_types=None, valencies=None,
        node_subtype_by_type=None, edge_subtype_by_family=None,
        node_type_distribution=None, edge_family_distribution=None
    ):
        self.num_nodes = num_nodes
        self.node_types = node_types  # 全局子类别分布（所有14个子类别）
        self.bond_types = bond_types  # 全局边类型分布（所有6个边类型）
        self.charge_types = charge_types
        self.valencies = valencies
        # 异质图：按类型/关系族分组的子类别分布
        self.node_subtype_by_type = node_subtype_by_type  # Dict[node_type_name, torch.Tensor] - 每个节点类型的子类别分布
        self.edge_subtype_by_family = edge_subtype_by_family  # Dict[edge_family_name, torch.Tensor] - 每个关系族的边子类别分布
        # 异质图：节点类型分布和关系族分布（用于初始化）
        self.node_type_distribution = node_type_distribution  # Dict[node_type_name, float] - 每个节点类型的比例分布
        self.edge_family_distribution = edge_family_distribution  # Dict[(src_type, dst_type), Dict[edge_family_name, float]] - 两个节点类型之间，每个关系族的比例分布


class RemoveYTransform:
    def __call__(self, data):
        data.y = torch.zeros((1, 0), dtype=torch.float)
        return data
