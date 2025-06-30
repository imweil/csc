import torch
import numpy as np


def pad_and_tensorize_adjacency_matrices(adj_matrices):
    if not adj_matrices:
        return torch.tensor([]), torch.tensor([])
    max_size = max(len(mat) for mat in adj_matrices)

    padded_matrices = []
    masks = []

    for mat in adj_matrices:
        num_nodes = len(mat)
        pad_size = max_size - num_nodes
        padded = np.pad(mat, ((0, pad_size), (0, pad_size)), mode='constant', constant_values=0)
        padded_matrices.append(padded)

        mask = torch.zeros(max_size)
        mask[:num_nodes] = 1
        masks.append(mask)

    padded_matrices_array = np.array(padded_matrices)

    tensor_adj = torch.from_numpy(padded_matrices_array)

    tensor_masks = torch.stack(masks)

    return tensor_adj, tensor_masks


def pad_and_tensorize_embeddings(embeddings_list):
    if not embeddings_list:
        return torch.tensor([])

    if not has_number(embeddings_list):
        return torch.tensor([])

    max_nodes = max(len(graph_emb) for graph_emb in embeddings_list if graph_emb)

    for item in embeddings_list:
        if has_number(item):
            emb_dim = len(item[0])
            break

    padded_embeddings = []
    for graph_emb in embeddings_list:
        if graph_emb:
            pad_size = max_nodes - len(graph_emb)
            padded = np.pad(graph_emb, ((0, pad_size), (0, 0)), mode='constant', constant_values=0)
        else:
            padded = np.zeros((max_nodes, emb_dim))
        padded_embeddings.append(padded)

    padded_embeddings_array = np.array(padded_embeddings)

    tensor_emb = torch.from_numpy(padded_embeddings_array).float()

    return tensor_emb


def has_number(lst):
    for item in lst:
        if isinstance(item, list):
            if has_number(item):
                return True
        else:
            if isinstance(item, (int, float)):
                return True
    return False


if __name__ == "__main__":
    # 示例邻接矩阵列表
    adj_matrices = [
        [[1, 1], [1, 1]],
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
    ]

    # 调用函数
    tensor_adj, tensor_masks = pad_and_tensorize_adjacency_matrices(adj_matrices)

    print("Adjacency Tensor:")
    print(tensor_adj)
    print("Adjacency Tensor Shape:", tensor_adj.shape)

    print("\nMask Tensor:")
    print(tensor_masks)
    print("Mask Tensor Shape:", tensor_masks.shape)

# if __name__ == "__main__":
#     # 示例嵌入列表
#     embeddings_list = [
#         [[1, 2,4], [3, 4,5]],
#         [[1, 2,5], [3, 4,5], [5, 6,5]],
#
#     ]
#
#     # 调用函数
#     tensor_emb = pad_and_tensorize_embeddings(embeddings_list)
#
#     print("Embeddings Tensor:")
#     print(tensor_emb)
#     print("Embeddings Tensor Shape:", tensor_emb.shape)
