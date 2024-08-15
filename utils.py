import math
import torch
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.data import Data
from collections import defaultdict


# # 创建每个实体的邻居集合
# neighbors = defaultdict(lambda: defaultdict(set))
#
# for head, tail, relation in triplets:
#     neighbors[head][relation].add(tail)
#
# # 计算规范化系数并存储结果
# entity_neighbors = {}
# for entity, rel_dict in neighbors.items():
#     entity_neighbors[entity] = {}
#     for rel, tails in rel_dict.items():
#         norm_coeff = len(tails)
#         entity_neighbors[entity][rel] = (tails, norm_coeff)
#
#
# # 获取局部邻居信息和聚合计算
# def get_all_relation_neighbors_and_aggregate(batch_heads):
#     all_relation_neighbors = {}
#
#     for head in batch_heads:
#         if head in entity_neighbors:
#             rel_neighbors = {}
#             for relation, (neighbors_set, norm_coeff) in entity_neighbors[head].items():
#                 rel_neighbors[relation] = (list(neighbors_set), norm_coeff)
#             all_relation_neighbors[head] = rel_neighbors
#         else:
#             all_relation_neighbors[head] = {}
#
#     return all_relation_neighbors
#
#
# # 示例批处理输入
# batch_heads = [0, 2, 4]
#
# # 获取局部邻居信息和规范化系数
# all_relation_neighbors = get_all_relation_neighbors_and_aggregate(batch_heads)
#
# # 输出聚合邻居信息
# for head, rel_neighbors in all_relation_neighbors.items():
#     print(f'Entity {head}:')
#     for rel, (neighbors_list, norm_coeff) in rel_neighbors.items():
#         print(f'  Relation {rel}: Neighbors: {neighbors_list}, Norm Coeff: {norm_coeff}')
def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)


def edge_normalization(edge_type, edge_index, num_entity, num_relation):
    '''
    Edge normalization trick
    - one_hot: (num_edge, num_relation)
    - deg: (num_node, num_relation)
    - index: (num_edge)
    - deg[edge_index[0]]: (num_edge, num_relation)
    - edge_norm: (num_edge)
    '''
    device = edge_type.device
    one_hot = F.one_hot(edge_type, num_classes=num_relation).to(torch.float).to(device)
    deg = scatter_add(one_hot, edge_index[0], dim=0, dim_size=num_entity).to(device)
    edge_norm = 1 / deg[edge_index[0], edge_type]

    return edge_norm


def generate_sampled_graph_and_labels(num_rels, device):
    file_path = './benchmarks/MKG-Y/train2id.txt'
    file_path1 = './benchmarks/MKG-Y/valid2id.txt'
    file_path2 = './benchmarks/MKG-Y/test2id.txt'
    heads = []
    rels = []
    tails = []
    with open(file_path, 'r') as f:
        next(f)  # 跳过第一行（如果第一行是数据的数量）
        for line in f:
            head, tail, relation = map(int, line.strip().split())
            heads.append(head)
            rels.append(relation)
            tails.append(tail)

    with open(file_path1, 'r') as f:
        next(f)  # 跳过第一行（如果第一行是数据的数量）
        for line in f:
            head, tail, relation = map(int, line.strip().split())
            heads.append(head)
            rels.append(relation)
            tails.append(tail)
    with open(file_path2, 'r') as f:
        next(f)  # 跳过第一行（如果第一行是数据的数量）
        for line in f:
            head, tail, relation = map(int, line.strip().split())
            heads.append(head)
            rels.append(relation)
            tails.append(tail)

    src = torch.tensor(heads, dtype=torch.long).to(device)
    dst = torch.tensor(tails, dtype=torch.long).to(device)
    rels = torch.tensor(rels, dtype=torch.long).to(device)
    uniq_entity, edges = torch.unique(torch.cat((src, dst)), return_inverse=True)
    src, dst = torch.reshape(edges, (2, -1))
    edge_index = torch.stack((src, dst))
    edge_type = rels
    data = Data(edge_index=edge_index)
    data.entity = uniq_entity
    data.edge_type = edge_type
    data.edge_norm = edge_normalization(edge_type, edge_index, len(uniq_entity), num_rels)

    return data

