import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
from datasets.CUB200 import CUB200
from models.resnet import resnet152
import config




if __name__ == '__main__':
    train_CUB = CUB200(mode='train')
    test_CUB = CUB200(mode='val')
    train_dataloader = DataLoader(train_CUB, batch_size=4, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)
    test_dataloader = DataLoader(test_CUB, batch_size=1, shuffle=False, num_workers=2, pin_memory=True, drop_last=False)

    net = resnet152(pretrained=True)
    net.train()
    net = net.cuda()

    optimizer = optim.SGD(net.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    total_loss = 0
    iter_num = 0
    corrects = 0

    data = iter(train_dataloader)
    for i in range(config.max_epoch):
        try:
            image, label = next(data)
        except:
            data = iter(train_dataloader)
            image, label = next(data)

        iter_num += 1
        image, label = image.cuda(), label.cuda()
        output = net(image)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if iter_num % 10 == 0:
            print('epoch:{}, iter:{}, avg_loss:{}'.format(i, iter_num, total_loss / 10.0))
            total_loss = 0

        if iter_num % 500 == 0:
            net.eval()
            test_data = iter(test_dataloader)
            while 1:
                try:
                    image, label = next(test_data)
                except:
                    break
                image, label = image.cuda(), label.cuda()
                output = net(image)
                _, preds = torch.max(output, dim=1)
                corrects += torch.sum(preds == label.data)

            acc = float(corrects) / float(test_dataloader.__len__())
            print('acc = {}'.format(acc))
            net.train()





