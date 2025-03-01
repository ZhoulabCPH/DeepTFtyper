import torch
import torch.nn as nn
from torchvision import transforms as tf
from torch.utils.data import Dataset, DataLoader

import os
from tqdm import tqdm
from PIL import Image
from ctran import ctranspath

import warnings
warnings.filterwarnings("ignore")

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


class PatchDataset(Dataset):
    def __init__(self, root, patch_paths):
        self.root = root
        self.patch_paths = patch_paths
        self.transform = tf.Compose([
            tf.Resize((224, 224)),
            tf.ToTensor(),
            tf.Normalize(mean=mean, std=std)
        ])

    def __len__(self):
        return len(self.patch_paths)

    def __getitem__(self, item):
        image_path = self.root + '/' + self.patch_paths[item]
        image = Image.open(image_path)
        image = self.transform(image)

        return image


@torch.no_grad()
def extract_features(root, epoch, epochs, patch_paths, models, device):
    dataset = PatchDataset(root, patch_paths)
    dataloader = DataLoader(dataset=dataset, batch_size=256, shuffle=False)
    with tqdm(total=len(dataloader)) as _tqdm:
        _tqdm.set_description('epoch: {}/{}'.format(epoch + 1, epochs))

        features = [[] for i in range(len(models))]
        for image in dataloader:
            inputs = image.to(device)
            for i, model in enumerate(models):
                outputs = model(inputs)
                features[i].extend(outputs.cpu())
            _tqdm.update(1)

        return features

def extract(slide_root, save_roots, models, device):
    slide_names = os.listdir(slide_root)
    for i, slide_name in enumerate(slide_names):
        patch_paths = os.listdir(slide_root + slide_name)
        features = extract_features(slide_root + slide_name, i, len(slide_names), patch_paths, models, device) # 提取特征
        patch_names = []
        for patch in patch_paths:
            name = patch.split('.jpeg')[0]
            patch_names.append(name)

        for j in range(len(features)):
            feature = torch.stack(features[j])
            datas = {'features': feature, 'patch_names': patch_paths}
            torch.save(datas, save_roots[j] + slide_name + '.pkl')


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ctranspath
    model_path = ''
    model_ctranspath = ctranspath()
    model_ctranspath.head = nn.Identity()
    model_ctranspath.load_state_dict(torch.load(model_path)['model'], strict=True)
    model_ctranspath.eval()
    model_ctranspath = model_ctranspath.to(device)

    # 提取特征
    models = [model_ctranspath]
    slide_root = ''
    save_root = ''
    save_roots = [save_root]
    extract(slide_root, save_roots, models, device)