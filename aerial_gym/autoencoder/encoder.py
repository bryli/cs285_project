import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .vgg_new import VGG_backbone
import aerial_gym.autoencoder.my_custom_transforms as mtr
import torchvision
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import glob
import os
from os.path import join
from torch.utils.data import DataLoader
from tqdm import tqdm


def init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(m.bias, -bound, bound)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

class Single_Stream(nn.Module):
    def __init__(self, in_channel=3, vgg_path='./model/vgg16_feat.pth'):
        super(Single_Stream, self).__init__()
        self.backbone = VGG_backbone(in_channel=in_channel, pre_train_path=vgg_path)
        self.toplayer = nn.Sequential(
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(512, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)  ###
        )
        init_weight(self.toplayer)

    def forward(self, input):
        l1 = self.backbone.conv1(input)
        l2 = self.backbone.conv2(l1)
        l3 = self.backbone.conv3(l2)
        l4 = self.backbone.conv4(l3)
        l5 = self.backbone.conv5(l4)
        l6 = self.toplayer(l5)
        return l6

class MyNet(nn.Module):
    def __init__(self, vgg_path='./model/vgg16_feat.pth'):
        super(MyNet, self).__init__()

        # Main-streams
        self.main_stream = Single_Stream(in_channel=3, vgg_path=vgg_path)

    def forward(self, input):
        dep = input
        dep = dep.repeat(1, 3, 1, 1)
        feat = self.main_stream(dep)
        return feat

    def get_input(self, sample_batched):
        dep =sample_batched['depth'].cuda()
        return dep

class RgbdSodDataset(Dataset):
    def __init__(self, datasets=None, transform=None, max_num=0, if_memory=False, loading_from_list=False, input_list=None):
        super().__init__()
        self.loading_from_list = loading_from_list
        self.input_list = input_list
        if not loading_from_list:
            if not isinstance(datasets, list):  datasets = [datasets]
            self.imgs_list, self.gts_list, self.depths_list = [], [], []

            for dataset in datasets:
                ids = sorted(glob.glob(os.path.join(dataset, 'RGB', '*.jpg')))
                ids = [os.path.splitext(os.path.split(id)[1])[0] for id in ids]
                for id in ids:
                    self.imgs_list.append(os.path.join(dataset, 'RGB', id + '.jpg'))
                    self.gts_list.append(os.path.join(dataset, 'GT', id + '.png'))
                    self.depths_list.append(os.path.join(dataset, 'depth', id + '.png'))

            if max_num != 0 and len(self.imgs_list) > abs(max_num):
                indices = random.sample(range(len(self.imgs_list)), max_num) if max_num > 0 else range(abs(max_num))
                self.imgs_list = [self.imgs_list[i] for i in indices]
                self.gts_list = [self.gts_list[i] for i in indices]
                self.depths_list = [self.depths_list[i] for i in indices]

            self.transform, self.if_memory = transform, if_memory

            if if_memory:
                self.samples = []
                for index in range(len(self.imgs_list)):
                    self.samples.append(self.get_sample(index))
        else:
            self.transform, self.if_memory = transform, if_memory
            if input_list is not None:
                if not isinstance(input_list, list):
                    raise Exception("Dataset input is not a list")
    def __len__(self):
        if not self.loading_from_list:
            return len(self.imgs_list)
        elif self.loading_from_list:
            return len(self.input_list)

    def __getitem__(self, index):
        if not self.loading_from_list:
            if self.if_memory:
                return self.transform(self.samples[index].copy()) if self.transform != None else self.samples[index].copy()
            else:
                return self.transform(self.get_sample(index)) if self.transform != None else self.get_sample(index)
        else:
            put_into_dic = {'depth':self.input_list[index]}
            return self.transform(put_into_dic)

    def get_sample(self, index):
        img = np.array(Image.open(self.imgs_list[index]).convert('RGB'))
        gt = np.array(Image.open(self.gts_list[index]).convert('L'))
        depth = np.array(Image.open(self.depths_list[index]).convert('L'))
        sample = {'img': img, 'gt': gt, 'depth': depth}

        sample['meta'] = {'id': os.path.splitext(os.path.split(self.gts_list[index])[1])[0]}
        sample['meta']['source_size'] = np.array(gt.shape[::-1])
        sample['meta']['img_path'] = self.imgs_list[index]
        sample['meta']['gt_path'] = self.gts_list[index]
        sample['meta']['depth_path'] = self.depths_list[index]
        return sample

class Autoencoder():
    def __init__(self, VGG_checkpoint='./model/vgg16_feat.pth', encoder_checkpoint='./model/best_depth.pth'):
        self.model = MyNet(VGG_checkpoint).cuda()
        self.pretrained= torch.load(encoder_checkpoint)
        pretrained_dict = {k: v for k, v in self.pretrained['model'].items() if k in self.model.state_dict()}
        self.model.state_dict().update(pretrained_dict)
        self.transform_test = torchvision.transforms.Compose([mtr.Resize((224, 224)), mtr.ToTensor(),
                                                         mtr.Normalize(mean=[0.485, 0.456, 0.406],
                                                                       std=[0.229, 0.224, 0.225], elems_do=['img'])])


    def forward(self, inputs):
        self.val_set = RgbdSodDataset(datasets=None, transform=self.transform_test, max_num=0,
                                      if_memory=False, loading_from_list=True, input_list=inputs)
        # val_set = val_set.get_sample(42)
        self.test_loader = DataLoader(self.val_set, batch_size=8, shuffle=False, pin_memory=True)
        for i, sample_batched in enumerate(tqdm(self.test_loader)):
            input = self.model.get_input(sample_batched)
            with torch.no_grad():
                output = self.model(input)
                if i == 0:
                    outputs = output
                else:
                    outputs = torch.cat((outputs, output),0)
        outputs = torch.squeeze(outputs) # (Batch, 7, 7)
        outputs = torch.flatten(outputs, start_dim=1) # (Batch, 49)
        outputs = outputs.cuda()
        return outputs
if __name__ == "__main__":
    # Reading dataset locally, nothing to do with online simulation
    datasets_path = './dataset/TestingSet/'
    # test_datasets = ['SSD', 'DES', 'LFSD', 'NJU2K_TEST', 'SIP', 'STERE']
    test_datasets = ['SSD']
    transform_test = torchvision.transforms.Compose([mtr.Resize((224,224)), mtr.ToTensor(),
                                                     mtr.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225], elems_do=['img'])])

    test_loaders = []
    for test_dataset in test_datasets:
        val_set = RgbdSodDataset(datasets_path + test_dataset, transform=transform_test)
        val_set = val_set.get_sample(42)
    # Generating an input list from one sample from the offine dataset
    image = val_set['depth']
    images = [image for _ in range(50)]

    # Here comes the important part
    # Pass dic of VGG pth file and pretained model
    auto = Autoencoder(encoder_checkpoint = "H:\\225Final\D3NetBenchmark-master\snapshot\\2023-12-03-22-23-47_DepthNet\\best.pth")

    # Pass the input link here
    opt = auto.forward(images)
    # Output's shape is like(Batch, 49), Batch = len(input)
    print(opt)





