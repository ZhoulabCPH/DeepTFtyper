import torch
from torch.nn.functional import one_hot
from sklearn.metrics import accuracy_score, roc_curve, auc

import sys
import numpy as np
from tqdm import tqdm


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = torch.nn.CrossEntropyLoss(weight=self.weight)(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        return focal_loss


class Trainer(object):
    def train(self, dataloader, model, optimizer, device, epoch, num_epoch):
        with tqdm(total=len(dataloader)) as _tqdm:
            _tqdm.set_description('epoch: [{}/{}] - train'.format(epoch + 1, num_epoch))

            focal_loss = FocalLoss(weight=torch.tensor([1.2, 1.0, 1.2, 0.8], device=device))
            for data in dataloader:
                optimizer.zero_grad()
                Y, S, G, SG = data['y'], data['s'], data['graph'], data['sparse_graph']
                outputs, num, d_loss = [], 0, 0.
                for y, graph, sparse_graph in zip(Y, G, SG):
                    x, e1, e2 = graph.x.unsqueeze(0).to(device), graph.edge_index.to(device), sparse_graph.edge_index.to(device)
                    output = model.forward(x, e1, e2)
                    outputs.append(output)
                    d_loss += model.d_loss
                    num += 1
                outputs = torch.cat(outputs)
                targets = torch.tensor(S, dtype=torch.float, device=device)
                loss = focal_loss(outputs, targets) + d_loss / num

                # 更新参数
                loss.backward()
                optimizer.step()
                _tqdm.update(1)
            info = {'lr': '{:.12f}'.format(optimizer.state_dict()['param_groups'][0]['lr'])}
            _tqdm.set_postfix(**info)


class Evaluator(object):
    @torch.no_grad()
    def eval(self, dataloader, model, device, name):
        with tqdm(total=len(dataloader), file=sys.stdout) as _tqdm:
            _tqdm.set_description('-----> ' + name)

            pred_scores, true_labels = [], []
            for data in dataloader:
                outputs = []
                Y, S, G, SG = data['y'], data['s'], data['graph'], data['sparse_graph']
                for y, graph, sparse_graph in zip(Y, G, SG):
                    x, e1, e2 = graph.x.unsqueeze(0).to(device), graph.edge_index.to(
                        device), sparse_graph.edge_index.to(device)
                    output = model.forward(x, e1, e2)
                    outputs.append(output)
                outputs = torch.cat(outputs)
                pred_scores.append(torch.softmax(outputs, dim=-1))
                true_labels.extend(Y)
                _tqdm.update(1)
            pred_scores = torch.cat(pred_scores).cpu().detach().numpy()
            true_label_onehot = one_hot(torch.tensor(true_labels, dtype=torch.long), pred_scores.shape[1]).numpy()

            # auc
            fpr, tpr, _ = roc_curve(true_label_onehot[:, 0], pred_scores[:, 0])
            a_auc = auc(fpr, tpr)

            fpr, tpr, _ = roc_curve(true_label_onehot[:, 1], pred_scores[:, 1])
            n_auc = auc(fpr, tpr)

            fpr, tpr, _ = roc_curve(true_label_onehot[:, 2], pred_scores[:, 2])
            p_auc = auc(fpr, tpr)

            fpr, tpr, _ = roc_curve(true_label_onehot[:, 3], pred_scores[:, 3])
            y_auc = auc(fpr, tpr)

            info = {'A_auc': '{:.2f}'.format(a_auc), 'N_auc': '{:.2f}'.format(n_auc), 'P_auc': '{:.2f}'.format(p_auc),
                    'Y_auc': '{:.2f}'.format(y_auc)}
            _tqdm.set_postfix(**info)

            return a_auc, n_auc, p_auc, y_auc