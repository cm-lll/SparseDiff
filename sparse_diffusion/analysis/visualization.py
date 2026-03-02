import os
import os.path as osp

try:
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem
    from rdkit.Geometry import Point3D
    from rdkit import RDLogger
    import rdkit.Chem
    from sparse_diffusion.metrics.molecular_metrics import Molecule, SparseMolecule
    _RDKIT_AVAILABLE = True
except ImportError:
    Chem = Draw = AllChem = Point3D = RDLogger = None
    Molecule = SparseMolecule = None
    _RDKIT_AVAILABLE = False

import imageio
import networkx as nx
import numpy as np
import wandb
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


class Visualizer:
    def __init__(self, dataset_infos):
        self.dataset_infos = dataset_infos
        self.is_molecular = self.dataset_infos.is_molecular
        # 检查是否为异质图
        self.is_heterogeneous = getattr(dataset_infos, 'heterogeneous', False)

        if self.is_molecular:
            self.remove_h = dataset_infos.remove_h
        
        # 异质图相关：获取节点类型和关系类型的解码器
        if self.is_heterogeneous:
            # 尝试从dataset_infos获取解码器
            self.node_type_decoder = getattr(dataset_infos, 'node_type_decoder', None)
            self.relation_type_decoder = getattr(dataset_infos, 'relation_type_decoder', None)
            self.node_subtype_decoder = getattr(dataset_infos, 'node_subtype_decoder', None)
            
            # 节点子标签：优先从 dataset_infos.node_subtype_decoder / node_subtype_names 获取
            if self.node_subtype_decoder is None and hasattr(dataset_infos, 'node_subtype_names') and dataset_infos.node_subtype_names:
                self.node_subtype_decoder = {i: n for i, n in enumerate(dataset_infos.node_subtype_names)}
            if self.node_subtype_decoder is None:
                vocab_path = getattr(dataset_infos, 'vocab_path', None)
                if vocab_path is None and hasattr(dataset_infos, 'datamodule') and hasattr(dataset_infos.datamodule, 'inner'):
                    vocab_path = osp.join(dataset_infos.datamodule.inner.processed_dir, 'vocab.json')
                if vocab_path and osp.exists(vocab_path):
                    try:
                        import json
                        with open(vocab_path, 'r') as f:
                            vocab = json.load(f)
                        if 'node_subtype_names' in vocab:
                            self.node_subtype_decoder = {i: name for i, name in enumerate(vocab['node_subtype_names'])}
                    except Exception:
                        pass
            
            # 构建节点类型解码器（从type2id反向）
            if self.node_type_decoder is None:
                node_type2id = getattr(dataset_infos, 'node_type2id', {})
                if node_type2id:
                    self.node_type_decoder = {v: k for k, v in node_type2id.items()}
                else:
                    node_type_names = getattr(dataset_infos, 'node_type_names', [])
                    if node_type_names:
                        self.node_type_decoder = {i: name for i, name in enumerate(node_type_names)}
            
            # 构建关系类型解码器：优先 edge_label2id（子标签如 author_of:first_author），其次 edge_family2id（类别）
            if self.relation_type_decoder is None:
                edge_label2id = getattr(dataset_infos, 'edge_label2id', {})
                if edge_label2id:
                    self.relation_type_decoder = {int(v): k for k, v in edge_label2id.items()}
                else:
                    edge_family2id = getattr(dataset_infos, 'edge_family2id', {})
                    if edge_family2id:
                        self.relation_type_decoder = {v: k for k, v in edge_family2id.items()}

            # 预计算：节点类型→颜色索引、关系族→颜色索引（用于异质图着色）
            node_type_names = getattr(dataset_infos, 'node_type_names', [])
            edge_family_offsets = getattr(dataset_infos, 'edge_family_offsets', {})
            self._node_type_to_color = {t: i for i, t in enumerate(node_type_names)} if node_type_names else {}
            # 按 offset 排序关系族，保证顺序稳定
            fams_sorted = sorted(edge_family_offsets.keys(), key=lambda f: edge_family_offsets.get(f, 0))
            self._edge_family_to_color = {f: i for i, f in enumerate(fams_sorted)} if fams_sorted else {}

    def _get_type_sizes(self):
        """从 type_offsets 推断各节点类型的子类别数量（type_sizes）。"""
        type_offsets = getattr(self.dataset_infos, 'type_offsets', None) or {}
        total = len(getattr(self.dataset_infos, 'node_types', []))
        if total == 0 and hasattr(self.dataset_infos, 'output_dims') and self.dataset_infos.output_dims is not None:
            total = getattr(self.dataset_infos.output_dims, 'X', 0)
        sorted_offs = sorted(type_offsets.items(), key=lambda x: x[1])
        out = {}
        for i, (t, off) in enumerate(sorted_offs):
            next_off = sorted_offs[i + 1][1] if i + 1 < len(sorted_offs) else total
            out[t] = max(0, next_off - off)
        return out

    def _edge_label_to_family(self, edge_label_id):
        """从全局 edge_label ID 推断关系族名称。"""
        eo = getattr(self.dataset_infos, 'edge_family_offsets', None) or {}
        if not eo or edge_label_id <= 0:
            return None
        sorted_fams = sorted(eo.items(), key=lambda x: x[1])
        for fam, off in reversed(sorted_fams):
            if off <= edge_label_id:
                return fam
        return None

    def to_networkx(self, node, edge_index, edge_attr):
        """
        Convert graphs to networkx graphs
        node_list: the nodes of a batch of nodes (bs x n)
        adjacency_matrix: the adjacency_matrix of the molecule (bs x n x n)
        """
        if self.is_heterogeneous:
            # 异质图：使用有向图，节点/边带类型名与 color 索引（按类型、关系族着色）
            graph = nx.DiGraph()
            type_offsets = getattr(self.dataset_infos, 'type_offsets', {})
            node_type_names = getattr(self.dataset_infos, 'node_type_names', [])
            type_sizes = self._get_type_sizes()

            if len(node) == 0:
                return graph
            for i in range(len(node)):
                node_subtype_id = int(node[i])
                node_type_name = None
                if type_offsets and type_sizes:
                    for t, off in type_offsets.items():
                        sz = type_sizes.get(t, 0)
                        if off <= node_subtype_id < off + sz:
                            node_type_name = t
                            break
                # 标签优先使用子类别名称（子标签），其次类别名称
                if self.node_subtype_decoder and node_subtype_id in self.node_subtype_decoder:
                    display_label = self.node_subtype_decoder[node_subtype_id]
                elif node_type_name is not None:
                    display_label = node_type_name
                else:
                    display_label = f"Subtype_{node_subtype_id}"
                if node_type_name is None:
                    node_type_name = display_label
                # color_val = 节点类型在 node_type_names 中的索引，用于按类型着色
                node_type_id = self._node_type_to_color.get(node_type_name, 0)
                graph.add_node(
                    i, number=i, symbol=node_subtype_id, color_val=node_type_id,
                    node_type=node_type_name, label=display_label
                )

            # 边：edge_index 形状 (2, E)，与 edge_attr (E,) 逐条对应
            if edge_index.size == 0:
                return graph
            ei = edge_index.T if edge_index.shape[0] == 2 else edge_index  # (E, 2)
            ea = np.ravel(edge_attr)
            edges_added = 0
            edges_skipped = 0
            for i in range(ei.shape[0]):
                edge_label_id = int(ea[i]) if i < ea.size else 0
                if edge_label_id == 0:
                    edges_skipped += 1
                    continue
                u, v = int(ei[i, 0]), int(ei[i, 1])
                edges_added += 1
                # 边标签优先使用子类别（子标签，如 author_of:first_author），其次关系族（类别）
                if self.relation_type_decoder and edge_label_id in self.relation_type_decoder:
                    relation_type_name = self.relation_type_decoder[edge_label_id]
                else:
                    relation_type_name = self._edge_label_to_family(edge_label_id)
                if relation_type_name is None:
                    relation_type_name = f"Rel_{edge_label_id}"
                # color = 关系族在排序列表中的索引（子标签如 author_of:first_author 取前半段 author_of）
                fam_key = relation_type_name.split(':')[0] if ':' in str(relation_type_name) else relation_type_name
                edge_family_id = self._edge_family_to_color.get(fam_key, 0)
                graph.add_edge(
                    u, v,
                    color=edge_family_id, weight=2,
                    relation_type=relation_type_name, label=relation_type_name
                )
            # 调试：如果添加的边数为0，打印警告
            if edges_added == 0 and ei.shape[0] > 0:
                print(f"  [WARN] to_networkx: 有 {ei.shape[0]} 条边，但全部被跳过（edge_label_id=0），添加了 0 条边")
                print(f"    edge_attr 值: {ea[:min(10, len(ea))]}")
                print(f"    edge_index 前5条: {ei[:min(5, ei.shape[0])]}")
        else:
            # 同质图：使用无向图
            graph = nx.Graph()

            for i in range(len(node)):
                graph.add_node(i, number=i, symbol=node[i], color_val=node[i])

            for i, edge in enumerate(edge_index.T):
                edge_type = edge_attr[i]
                graph.add_edge(edge[0], edge[1], color=edge_type, weight=3 * edge_type)

        return graph

    def visualize_non_molecule(
        self, graph, pos, path, iterations=100, node_size=100, largest_component=False
    ):
        if largest_component:
            # 对于有向图，使用弱连通分量
            if isinstance(graph, nx.DiGraph):
                CGs = [graph.subgraph(c) for c in nx.weakly_connected_components(graph)]
            else:
                CGs = [graph.subgraph(c) for c in nx.connected_components(graph)]
            CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
            graph = CGs[0]

        # Plot the graph structure with colors
        if pos is None:
            G_layout = graph.to_undirected() if isinstance(graph, nx.DiGraph) else graph
            pos = nx.spring_layout(G_layout, iterations=iterations, seed=42)

        # 异质图：按节点类型、关系族着色，标签为类型/关系名
        if self.is_heterogeneous and graph.number_of_nodes() > 0:
            node_colors = [graph.nodes[n].get('color_val', 0) for n in graph.nodes()]
            node_labels = {n: graph.nodes[n].get('label', str(n)) for n in graph.nodes()}
            n_max = max(node_colors) if node_colors else 0
            node_cmap = plt.cm.tab10 if n_max < 10 else plt.cm.Set3
            node_vmin, node_vmax = -0.5, max(n_max, 0) + 0.5

            if len(graph.edges()) > 0:
                edge_colors = [graph.edges[e].get('color', 0) for e in graph.edges()]
                edge_labels = {e: graph.edges[e].get('label', '') for e in graph.edges()}
                e_max = max(edge_colors) if edge_colors else 0
                edge_cmap = plt.cm.Paired if e_max < 12 else plt.cm.Set3
                edge_vmin, edge_vmax = -0.5, max(e_max, 0) + 0.5
            else:
                edge_colors = "grey"
                edge_labels = {}
                edge_cmap = None
                edge_vmin = edge_vmax = 0
        else:
            # 同质图
            try:
                if isinstance(graph, nx.DiGraph):
                    laplacian = nx.normalized_laplacian_matrix(graph.to_undirected()).toarray()
                else:
                    laplacian = nx.normalized_laplacian_matrix(graph).toarray()
                w, U = np.linalg.eigh(laplacian)
                vmin, vmax = np.min(U[:, 1]), np.max(U[:, 1])
                m = max(np.abs(vmin), vmax)
                node_colors = U[:, 1]
                node_cmap = plt.cm.coolwarm
                node_vmin, node_vmax = -m, m
                node_labels = {}
            except Exception:
                node_colors = list(graph.nodes())
                node_cmap = plt.cm.coolwarm
                node_vmin, node_vmax = None, None
                node_labels = {}
            edge_colors = "grey"
            edge_labels = {}
            edge_cmap = None
            edge_vmin = edge_vmax = 0

        plt.figure(figsize=(12, 8))
        # 绘制节点
        nx.draw_networkx_nodes(
            graph, pos, node_size=node_size, node_color=node_colors,
            cmap=node_cmap, vmin=node_vmin, vmax=node_vmax, alpha=0.8,
        )
        # 绘制边
        if isinstance(graph, nx.DiGraph):
            nx.draw_networkx_edges(
                graph, pos, edge_color=edge_colors, alpha=0.6,
                arrows=True, arrowsize=10, arrowstyle='->',
                edge_cmap=edge_cmap if edge_cmap and isinstance(edge_colors, list) else None,
                edge_vmin=edge_vmin if isinstance(edge_colors, list) else None,
                edge_vmax=edge_vmax if isinstance(edge_colors, list) else None,
            )
        else:
            nx.draw_networkx_edges(graph, pos, edge_color=edge_colors, alpha=0.6)
        if node_labels:
            nx.draw_networkx_labels(graph, pos, node_labels, font_size=7)
        if edge_labels and len(edge_labels) < 50:
            nx.draw_networkx_edge_labels(graph, pos, edge_labels, font_size=4)

        plt.axis('off')
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close("all")

    def visualize(
        self, path: str, graphs: list, num_graphs_to_visualize: int, log="graph", local_rank=0
    ):
        # define path to save figures
        if not os.path.exists(path):
            os.makedirs(path)

        # visualize the final molecules
        num_graphs = max(graphs.batch) + 1
        num_graphs_to_visualize = min(num_graphs_to_visualize, num_graphs)
        
        # 跳过节点数 > 20 的大图可视化（可视化效果差且慢）
        MAX_NODES_FOR_VIS = 20
        graphs_to_vis = []
        for i in range(num_graphs_to_visualize):
            if hasattr(graphs, 'ptr') and len(graphs.ptr) > i + 1:
                num_nodes_i = int(graphs.ptr[i + 1].item() - graphs.ptr[i].item())
            else:
                node_mask_i = graphs.batch == i
                num_nodes_i = int(node_mask_i.sum().item())
            if num_nodes_i <= MAX_NODES_FOR_VIS:
                graphs_to_vis.append(i)
        
        print(f"Visualizing {len(graphs_to_vis)} graphs out of {num_graphs} (skipped {num_graphs_to_visualize - len(graphs_to_vis)} with >{MAX_NODES_FOR_VIS} nodes)")

        for i in graphs_to_vis:
            file_path = os.path.join(path, "graph_{}_{}.png".format(i, local_rank))
            node_mask = graphs.batch == i
            edge_mask = graphs.batch[graphs.edge_index[0]] == i

            if self.is_molecular:
                if not _RDKIT_AVAILABLE:
                    raise ImportError("rdkit is required for molecular visualization")
                # TODO: change graph lists to a list of PlaceHolders
                molecule = SparseMolecule(
                    node_types=graphs.node[node_mask].long(),
                    bond_index=graphs.edge_index[:, edge_mask].long() - graphs.ptr[i],
                    bond_types=graphs.edge_attr[edge_mask].long(),
                    atom_decoder=self.dataset_infos.atom_decoder,
                    charge=None,
                )
                mol = molecule.rdkit_mol
                try:
                    Draw.MolToFile(mol, file_path)
                except rdkit.Chem.KekulizeException:
                    print("Can't kekulize molecule")
            else:
                # 处理 edge_attr：如果是 one-hot 编码，转换为离散值
                edge_attr_vis = graphs.edge_attr[edge_mask]
                if edge_attr_vis.dim() > 1:
                    # one-hot 编码，转换为离散值
                    edge_attr_vis = edge_attr_vis.argmax(dim=-1)
                edge_attr_vis = edge_attr_vis.long().cpu().numpy()
                
                # 调试：检查 edge_attr 的值
                if i < 3:  # 只打印前3个图的调试信息
                    num_edges = edge_attr_vis.shape[0] if edge_attr_vis.size > 0 else 0
                    num_nonzero = (edge_attr_vis > 0).sum() if edge_attr_vis.size > 0 else 0
                    if edge_attr_vis.size > 0:
                        print(f"  [DEBUG] Graph {i}: {num_edges} edges, {num_nonzero} non-zero (edge_attr range: {edge_attr_vis.min()}-{edge_attr_vis.max()})")
                    else:
                        print(f"  [DEBUG] Graph {i}: {num_edges} edges, {num_nonzero} non-zero (edge_attr is empty)")
                
                graph = self.to_networkx(
                    node=graphs.node[node_mask].long().cpu().numpy(),
                    edge_index=(graphs.edge_index[:, edge_mask].long() - graphs.ptr[i])
                    .cpu()
                    .numpy(),
                    edge_attr=edge_attr_vis,
                )

                self.visualize_non_molecule(graph=graph, pos=None, path=file_path)

            if wandb.run is not None and log is not None:
                if i < 3:
                    print(f"Saving {file_path} to wandb")
                wandb.log({log: [wandb.Image(file_path)]}, commit=False)

    def visualize_chain(self, chain_path, batch_id, chain, local_rank):
        node_list = chain.node_list
        edge_index_list = chain.edge_index_list
        edge_attr_list = chain.edge_attr_list
        batch = chain.batch
        ptr = chain.ptr

        keep_chain = int(chain.batch.max() + 1)
        MAX_NODES_FOR_VIS = 20

        for k in range(keep_chain):
            # 跳过节点数 > 20 的大图 chain 可视化
            if len(node_list) > 0:
                node_mask_k = batch == k
                num_nodes_k = int(node_mask_k.sum().item())
                if num_nodes_k > MAX_NODES_FOR_VIS:
                    continue
            path = os.path.join(chain_path, f"molecule_{batch_id + k}_{local_rank}")
            if not os.path.exists(path):
                os.makedirs(path)

            # get the list for molecules
            if self.is_molecular:
                if not _RDKIT_AVAILABLE:
                    raise ImportError("rdkit is required for molecular visualization")
                mols = []
                for i in range(len(node_list)):
                    node_mask = batch == k
                    edge_mask = batch[edge_index_list[i][0]] == k
                    mol = SparseMolecule(
                        node_types=node_list[i][node_mask].long(),
                        bond_index=edge_index_list[i][:, edge_mask].long() - ptr[k],
                        bond_types=edge_attr_list[i][edge_mask].long(),
                        atom_decoder=self.dataset_infos.atom_decoder,
                        charge=None,
                    ).rdkit_mol
                    mols.append(mol)

                # find the coordinates of atoms in the final molecule
                final_molecule = mols[-1]
                AllChem.Compute2DCoords(final_molecule)
                coords = []
                for i, atom in enumerate(final_molecule.GetAtoms()):
                    positions = final_molecule.GetConformer().GetAtomPosition(i)
                    coords.append((positions.x, positions.y, positions.z))

                # align all the molecules
                for i, mol in enumerate(mols):
                    AllChem.Compute2DCoords(mol)
                    conf = mol.GetConformer()
                    for j, atom in enumerate(mol.GetAtoms()):
                        x, y, z = coords[j]
                        conf.SetAtomPosition(j, Point3D(x, y, z))

                save_paths = []
                num_frames = len(node_list)

                for frame in range(num_frames):
                    file_name = os.path.join(path, "fram_{}.png".format(frame))
                    Draw.MolToFile(
                        mols[frame], file_name, size=(300, 300), legend=f"Frame {frame}"
                    )
                    save_paths.append(file_name)

            else:
                graphs = []
                for i in range(len(node_list)):
                    node_mask = batch == k
                    edge_mask = batch[edge_index_list[i][0]] == k
                    # 处理 edge_attr：如果是 one-hot 编码，转换为离散值
                    edge_attr_vis = edge_attr_list[i][edge_mask]
                    if edge_attr_vis.dim() > 1:
                        # one-hot 编码，转换为离散值
                        edge_attr_vis = edge_attr_vis.argmax(dim=-1)
                    edge_attr_vis = edge_attr_vis.long().cpu().numpy()
                    
                    graph = self.to_networkx(
                        node=node_list[i][node_mask].long().cpu().numpy(),
                        edge_index=(
                            edge_index_list[i][:, edge_mask].long() - ptr[k]
                        )
                        .cpu()
                        .numpy(),
                        edge_attr=edge_attr_vis,
                    )
                    graphs.append(graph)

                # find the coordinates of atoms in the final molecule
                final_graph = graphs[-1]
                G_layout = final_graph.to_undirected() if isinstance(final_graph, nx.DiGraph) else final_graph
                final_pos = nx.spring_layout(G_layout, seed=0)

                save_paths = []
                num_frames = len(node_list)

                for frame in range(num_frames):
                    file_name = os.path.join(path, "fram_{}.png".format(frame))
                    self.visualize_non_molecule(
                        graph=graphs[frame], pos=final_pos, path=file_name
                    )
                    save_paths.append(file_name)

            print("\r{}/{} complete".format(k + 1, keep_chain), end="", flush=True)

            imgs = [imageio.v3.imread(fn) for fn in save_paths]
            gif_path = os.path.join(
                os.path.dirname(path), "{}_{}.gif".format(path.split("/")[-1], local_rank)
            )
            imgs.extend([imgs[-1]] * 10)
            imageio.mimsave(gif_path, imgs, subrectangles=True, duration=200)
            if wandb.run is not None:
                wandb.log(
                    {"chain": [wandb.Video(gif_path, caption=gif_path, format="gif")]}, commit=False
                )
                print(f"Saving {gif_path} to wandb")
                wandb.log(
                    {"chain": wandb.Video(gif_path, fps=8, format="gif")}, commit=False
                )

