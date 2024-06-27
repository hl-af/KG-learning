import numpy as np
import networkx as nx

# 构建关系超图测试
# 示例数据
triples = [
    ['head1', 'relation1', 'tail1'],
    ['head2', 'relation2', 'tail2'],
    ['head1', 'relation2', 'tail3'],
]

# 假设每个实体和关系都有一个嵌入向量
embeddings = {
    'head1': np.array([0.1, 0.2, 0.3]),
    'head2': np.array([0.4, 0.5, 0.6]),
    'relation1': np.array([0.7, 0.8, 0.9]),
    'relation2': np.array([1.0, 1.1, 1.2]),
    'tail1': np.array([1.3, 1.4, 1.5]),
    'tail2': np.array([1.6, 1.7, 1.8]),
    'tail3': np.array([1.9, 2.0, 2.1]),
}

# 拼接函数
def concatenate_features(h, r, t):
    h_emb = embeddings[h]
    r_emb = embeddings[r]
    t_emb = embeddings[t]
    return np.concatenate([h_emb, r_emb, t_emb])

# 构建关系超图
def build_relational_hypergraph(triples):
    hypergraph = nx.Graph()

    # 添加节点（实体和关系）
    entities = set([triple[0] for triple in triples] + [triple[2] for triple in triples])
    relations = set([triple[1] for triple in triples])
    hypergraph.add_nodes_from(entities, node_type='entity')
    hypergraph.add_nodes_from(relations, node_type='relation')

    # 添加超边（三元组）并计算特征
    for triple in triples:
        h, r, t = triple[0], triple[1], triple[2]
        hypergraph.add_edge(h, r, key=(h, r, t))
        hypergraph.add_edge(r, t, key=(h, r, t))
        feature_vector = concatenate_features(h, r, t)
        hypergraph.nodes[h]['feature'] = embeddings[h]
        hypergraph.nodes[r]['feature'] = embeddings[r]
        hypergraph.nodes[t]['feature'] = embeddings[t]

    return hypergraph

# 计算特征矩阵和邻接矩阵
def compute_matrices(hypergraph):
    nodes = list(hypergraph.nodes)
    node_indices = {node: idx for idx, node in enumerate(nodes)}

    # 计算特征向量的总长度
    feature_dim = len(next(iter(embeddings.values())))

    # 特征矩阵
    feature_matrix = np.zeros((len(nodes), feature_dim))
    for node in nodes:
        feature_vector = hypergraph.nodes[node].get('feature', np.zeros(feature_dim))
        feature_matrix[node_indices[node], :len(feature_vector)] = feature_vector

    # 邻接矩阵
    adjacency_matrix = nx.adjacency_matrix(hypergraph, nodelist=nodes).todense()

    return feature_matrix, adjacency_matrix, nodes

# 主函数
if __name__ == "__main__":
    # 构建关系超图
    hypergraph = build_relational_hypergraph(triples)

    # 计算特征矩阵和邻接矩阵
    feature_matrix, adjacency_matrix, nodes = compute_matrices(hypergraph)

    # 输出结果
    print("Nodes:", nodes)
    print("Feature Matrix:")
    print(feature_matrix)
    print("Adjacency Matrix:")
    print(adjacency_matrix)
