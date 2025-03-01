import torch
from torch_geometric.data import Data
from torch.nn.functional import normalize
from torch_geometric.utils import subgraph
from torch_geometric.utils import to_networkx

import os
import random
import numpy as np
import leidenalg as la
import concurrent.futures
from igraph import Graph as IGraph

import warnings
warnings.filterwarnings("ignore")


def get_coordinates(paths, patch_root):
    patch_paths = []
    coordinates = []
    for path in paths:
        coordinate = path.split('.')[0].split('_')
        patch_paths.append(patch_root.split('.')[0] + '/' + path)
        coordinates.append((int(coordinate[0]), int(coordinate[1])))
    return coordinates, patch_paths


def build_graph(coordinates, paths, features, limit_similarity=0.9):
    # 计算相似度
    norm = normalize(features, dim=-1)
    similarity_matrix = torch.mm(norm, norm.T)

    edges, sparse_edges = [], []
    for node, coordinate in enumerate(coordinates):
        # 获得当前节点邻居节点坐标
        x, y = coordinate[0], coordinate[1]
        neighbors = {(x + dx, y + dy) for dx in range(-1, 2) for dy in range(-1, 2) if (dx, dy) != (0, 0)}
        for neighbor in neighbors:
            if neighbor in coordinates:
                neighbor_node = coordinates.index(neighbor)
                similarity = similarity_matrix[node][neighbor_node]
                if similarity >= limit_similarity:
                    sparse_edges.append([node, neighbor_node])
                edges.append([node, neighbor_node])
    edge_index = torch.tensor(edges).T
    graph = Data(x=features, edge_index=edge_index, paths=paths, coordinates=coordinates)

    sparse_edge_index = torch.tensor(sparse_edges).T
    sparse_graph = Data(x=features, edge_index=sparse_edge_index, paths=paths, coordinates=coordinates)
    return graph, sparse_graph


def process_build_graph(slide_name):
    datas = torch.load(slide_root + slide_name)
    patch_root = slide_root + slide_name
    features, patch_names = datas['features'], datas['patch_names']
    coordinates, paths = get_coordinates(patch_names, patch_root)
    graph, sparse_graph = build_graph(coordinates, paths, features, s)

    graphs = {'graph': graph, 'sparse_graph': sparse_graph}
    torch.save(graphs, save_root + slide_name)
    print('build graph <' + slide_name + '> finish !')


if __name__ == '__main__':
    # 运行环境设置
    seed = 100
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    num_threads = os.cpu_count()

    # 构建
    similarity = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    for s in similarity:
        slide_root = ''
        save_root = ''
        if not os.path.exists(save_root):
            os.makedirs(save_root)

        slide_names = os.listdir(slide_root)
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            executor.map(process_build_graph, slide_names)