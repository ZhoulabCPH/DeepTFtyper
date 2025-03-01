import torch
from torch.backends import cudnn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

import os
import random
import logging
import numpy as np
from option import Options
from utils import Trainer, Evaluator
from models.model import Model, weight_init
from graph.graph_dataset import GraphDataset, collate

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    # 运行环境设置
    cudnn.deterministic = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed = 100
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

    # 超参数设置
    args = Options().parse()
    num_classes = args.num_classes
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.lr
    graph_root = args.graph_root
    external_graph_root = args.external_graph_root
    save_log_path = args.save_log_root + 'logs.log'
    train_info_path = args.train_info_path
    test_info_path = args.test_info_path
    val_info_path = args.val_info_path
    save_model_root = args.save_model_root
    if not os.path.exists(save_model_root):
        os.makedirs(save_model_root)
    to_test = args.to_test

    # 加载数据集
    train_dataset = GraphDataset(graph_root, train_info_path)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, collate_fn=collate, shuffle=True,
                                  drop_last=False)

    val_dataset = GraphDataset(graph_root, val_info_path)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, collate_fn=collate, shuffle=False)

    test_dataset = GraphDataset(graph_root, test_info_path)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, collate_fn=collate, shuffle=False)

    # 创建模型
    model = Model(num_classes=num_classes)
    model = model.to(device)
    model.apply(weight_init)

    # 定义优化器
    optimizer = torch.optim.NAdam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    scheduler = lr_scheduler.ExponentialLR(optimizer, 0.95)

    # 训练日志
    logging.basicConfig(level=logging.DEBUG, format='', datefmt='%a, %d %b %Y %H:%M:%S', filename=save_log_path,
                        filemode='w')

    # 定义训练器和评估器
    trainer = Trainer()
    evaluator = Evaluator()
    for epoch in range(num_epochs):
        # train
        model.train()
        trainer.train(train_dataloader, model, optimizer, device, epoch, num_epochs)

        # test
        model.eval()
        with torch.no_grad():
            logging.info('\n[epoch:' + str(epoch) + '] ')
            t_a, t_n, t_p, t_y = evaluator.eval(train_dataloader, model, device, 'train')
            i_a, i_n, i_p, i_y = evaluator.eval(test_dataloader, model, device, 'val')
            t = '[a_auc:' + str(int(t_a * 100)) + ' n_auc:' + str(int(t_n * 100)) + ' p_auc:' + str(
                int(t_p * 100)) + ' y_auc:' + str(int(t_y * 100)) + ']'
            i = '[a_auc:' + str(int(i_a * 100)) + ' n_auc:' + str(int(i_n * 100)) + ' p_auc:' + str(
                int(i_p * 100)) + ' y_auc:' + str(int(i_y * 100)) + ']'
            logging.info('[train] ' + t)
            logging.info('[test] ' + i)

            if to_test:
                e_a, e_n, e_p, e_y = evaluator.eval(test_dataloader, model, device, 'test')
                e = '[a_auc:' + str(int(e_a * 100)) + ' n_auc:' + str(int(e_n * 100)) + ' p_auc:' + str(
                    int(e_p * 100)) + ' y_auc:' + str(int(e_y * 100)) + ']'
                logging.info('[test] ' + e)

        # update
        scheduler.step()

        # save
        model_name = 'epoch_' + str(epoch)
        torch.save(model, save_model_root + model_name + '.pth')
