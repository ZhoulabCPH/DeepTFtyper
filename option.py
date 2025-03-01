import argparse


class Options():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--graph_root', type=str, default='')
        parser.add_argument('--external_graph_root', type=str, default='')
        parser.add_argument('--save_model_root', type=str, default='')
        parser.add_argument('--save_log_root', type=str, default='')

        parser.add_argument('--train_info_path', type=str, default='')
        parser.add_argument('--val_info_path', type=str, default='')
        parser.add_argument('--test_info_path', type=str, default='')

        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--batch_size', type=int, default=8)
        parser.add_argument('--num_classes', type=int, default=4)
        parser.add_argument('--num_epochs', type=int, default=100)
        parser.add_argument('--to_test', type=bool, default=False)
        self.parser = parser

    def parse(self):
        return self.parser.parse_args()
