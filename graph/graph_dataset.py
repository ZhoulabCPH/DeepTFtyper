import torch
import tables
from torch.utils.data import Dataset


def collate(batch):
    os_t = [b['os_t'] for b in batch]
    os_s = [b['os_s'] for b in batch]
    n = [b['n'] for b in batch]
    y = [b['y'] for b in batch]
    s = [b['s'] for b in batch]
    graph = [b['graph'] for b in batch]
    sparse_graph = [b['sparse_graph'] for b in batch]

    return {'os_times': os_t, 'os_states': os_s, 'name': n, 'y': y, 's': s, 'graph': graph, 'sparse_graph': sparse_graph}


class GraphDataset(Dataset):
    def __init__(self, graph_dataset_root, info_path, need_os=False):
        self.graph_dataset_root = graph_dataset_root
        with tables.open_file(info_path, 'r') as f:
            self.labels = f.root['labels'].read()
            slide_names = f.root['slide_names'].read()
            self.slide_names = [tmp.decode() for tmp in slide_names]
            self.soft_labels = f.root['soft_labels'].read()

            if need_os:
                self.os_times = f.root['os_times'].read()
                self.os_states = f.root['os_states'].read()
            else:
                self.os_times = f.root['labels'].read()
                self.os_states = f.root['labels'].read()
            f.close()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        sample = {}
        sample['os_t'] = self.os_times[item]
        sample['os_s'] = self.os_states[item]
        sample['n'] = self.slide_names[item]
        sample['y'] = self.labels[item]
        sample['s'] = self.soft_labels[item]
        graphs = torch.load(self.graph_dataset_root + self.slide_names[item] + '.pkl')
        sample['graph'] = graphs['graph']
        sample['sparse_graph'] = graphs['sparse_graph']

        return sample