import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class CIFAR10Net(nn.Module):
    def __init__(self):
        super().__init__()
        '''
        nn.Sequential()可以看作是一种特殊的list，它可以打包各种nn.Module模块，
        并让输入按照打包顺序通过各个模块

        nn.ReLU()是激活层（激活函数），对输入逐元素计算，y_i = max(x_i, 0)
        nn.ReLU(inplace=True)中inplace=True表示直接让输出覆盖到输入的内存（显存）地址上
        这样可以节约内存（显存）
        '''
        self.conv_fc = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            # [batch_size, 3, 32, 32] -> [batch_size, 64, 32, 32]

            nn.MaxPool2d(kernel_size=2, stride=2),
            # [batch_size, 64, 32, 32] -> [batch_size, 64, 16, 16]

            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            # [batch_size, 64, 16, 16] -> [batch_size, 128, 16, 16]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [batch_size, 128, 16, 16] -> [batch_size, 128, 8, 8]

            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [batch_size, 128, 16, 16] -> [batch_size, 128, 8, 8]

            nn.ReLU(inplace=True),

            nn.Flatten(),
            nn.Linear(in_features=128 * 4 * 4, out_features=10)
        )

    def forward(self, x: torch.Tensor):
        return self.conv_fc(x)

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')
    parser.add_argument('-data_dir', default='./data_dir/CIFAR10', help='CIFAR10 dataset path')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=32, type=int, help='batch size')
    parser.add_argument('-epochs', default=10, type=int, help='number of total epochs to run')
    parser.add_argument('-j', default=2, type=int, help='number of data loading workers (default: 2)')
    parser.add_argument('-lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('-log_dir', default='./logs/logs_CNN_CIFAR10', help='path for saving tensorboard logs')
    args = parser.parse_args()

    print(args)

    train_set = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, transform=torchvision.transforms.ToTensor(), download=True)
    test_set = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, transform=torchvision.transforms.ToTensor(), download=True)
    train_dataloader = DataLoader(dataset=train_set, batch_size=args.b, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(dataset=test_set, batch_size=args.b, shuffle=False, drop_last=False)

    net = CIFAR10Net()
    net.to(args.device)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    iterations = 0

    writer = SummaryWriter(args.log_dir)  # 初始化tensorboard的日志写入者

    for epoch in range(args.epochs):
        # 训练
        for img, label in train_dataloader:
            img = img.to(args.device)  # 注意，tensor的to函数不是原地操作，不是对原始数据进行移动，而是返回一个在device上的新数据
            label = label.to(args.device)
            optimizer.zero_grad()  # 清零梯度
            label_predict = net(img)  # 网络的输出标签，也就是预测标签
            acc_train_batch = (label_predict.argmax(dim=1) == label).float().mean().item()  # 训练batch上的正确率
            # item函数返回一个标量tensor的数值

            loss = F.cross_entropy(label_predict, label)  # 计算损失
            loss.backward()  # 损失反向传播
            optimizer.step()  # 进行一次梯度下降
            iterations += 1
            if iterations % 64 == 0:
                loss = loss.item()
                print(f'iterations={iterations}, loss={loss}, acc_train_batch={acc_train_batch}')
                writer.add_scalar('train_batch_loss', loss, iterations)

        # 测试
        with torch.no_grad():
            # 在no_grad模式下，pytorch的计算不会建立计算图，不需要保存反向传播所需要的中间变量，速度更快
            correct_sum = 0  # 预测正确的总数
            for img, label in test_dataloader:
                img = img.to(args.device)
                label = label.to(args.device)
                label_predict = net(img)
                correct_sum += (label_predict.argmax(dim=1) == label).float().sum().item()

            acc_test = correct_sum / test_dataloader.__len__()
            print(args)
            print(f'epoch={epoch}, acc_test={acc_test}')
            writer.add_scalar('test_acc', acc_test, epoch)








