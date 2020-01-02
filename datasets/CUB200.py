import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import config
from datasets.transform import *
from PIL import Image



class CUB200(Dataset):
    def __init__(self, mode='train'):
        super(CUB200, self).__init__()
        self.mode = mode
        self.imgsname = []
        self.label = []

        Fsplit = open('./CUB_200_2011/train_test_split.txt', 'r')
        Fimgname = open('./CUB_200_2011/images.txt', 'r')
        Flabel = open('./CUB_200_2011/image_class_labels.txt', 'r')

        if self.mode == 'train':
            for i in Fsplit.readlines():
                img = Fimgname.readline()
                lb = Flabel.readline()
                if i.strip().split(' ')[1] == '1':
                    self.imgsname.append(img.strip().split(' ')[1])
                    self.label.append(lb.strip().split(' ')[1])

        elif self.mode == 'val':
            for i in Fsplit.readlines():
                img = Fimgname.readline()
                lb = Flabel.readline()
                if i.strip().split(' ')[1] == '0':
                    self.imgsname.append(img.strip().split(' ')[1])
                    self.label.append(lb.strip().split(' ')[1])

        else:
            print('*****************************\n wrong mode! \n***************************')


        self.totensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(config.mean, config.std)
        ])

        self.transform = Compose([
            resize(config.size),
            HorizontalFlip(0.5)
        ])

    def __getitem__(self, idx):
        imgpth = './CUB_200_2011/images/' + self.imgsname[idx]

        label = int(self.label[idx]) - 1
        img = Image.open(imgpth)

        if self.mode == 'train':
            im_lb = dict(im = img, lb = label)
            im_lb = self.transform(im_lb)
            img, label = im_lb['im'], im_lb['lb']

        imgs = self.totensor(img)

        return imgs, label


    def __len__(self):
        return len(self.imgsname)








