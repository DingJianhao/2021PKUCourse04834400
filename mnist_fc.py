import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


train_set = torchvision.datasets.MNIST(root='./data_dir', train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_set = torchvision.datasets.MNIST(root='./data_dir', train=False, transform=torchvision.transforms.ToTensor(), download=True)

print('训练集图片数量', train_set.__len__())
print('测试集图片数量', test_set.__len__())
img, label = train_set[0]
print(img.shape, label)

# # 展示图片
# to_pil = torchvision.transforms.ToPILImage()
# to_pil(img).show()

# 定义数据集加载器
from torch.utils.data import DataLoader
batch_size = 64
train_dataloader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataloader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, drop_last=False)

for img, label in train_dataloader:
    print(img.shape, label.shape)
    break

class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(28 * 28, 10)  # 输入是1 * 28 * 28，设置为in_features；一共有10类，因此out_features=10
    def forward(self, x: torch.Tensor):
        x = x.view(x.shape[0], -1)  # 将x.shape从[N, 1, 28, 28]变换为[N, 1 * 28 * 28]
        return self.fc(x)

# 运行的设备，'cpu'或'cuda:0'、'cuda:1'等。'cuda:0'表示0号显卡
device = 'cpu'
# 新建网络。网络默认是在CPU上
net = MNISTNet()
# 将网络移动到device上。注意，nn.Module的to函数是一个原地操作，也就是会将调用者移动到device上
net.to(device)
# 使用SGD优化器；优化的参数是net.parameters()，也就是net的所有可训练参数；学习率1e-3
optimizer = torch.optim.SGD(net.parameters(), lr=1e-3)
# 训练max_epoch轮，也就是遍历这么多次数据集
max_epoch = 100
# 总的迭代次数
iterations = 0
for epoch in range(max_epoch):
    # 训练
    for img, label in train_dataloader:
        img = img.to(device)  # 注意，tensor的to函数不是原地操作，不是对原始数据进行移动，而是返回一个在device上的新数据
        label = label.to(device)
        optimizer.zero_grad()  # 清零梯度
        label_predict = net(img)  # 网络的输出标签，也就是预测标签
        acc_train_batch = (label_predict.argmax(dim=1) == label).float().mean().item()  # 训练batch上的正确率
        #item函数返回一个标量tensor的数值

        loss = F.cross_entropy(label_predict, label)  # 计算损失
        loss.backward()  # 损失反向传播
        optimizer.step()  # 进行一次梯度下降
        iterations += 1
        if iterations % 1024 == 0:
            print(f'iterations={iterations}, loss={loss.item()}, acc_train_batch={acc_train_batch}')

    # 测试
    with torch.no_grad():
        # 在no_grad模式下，pytorch的计算不会建立计算图，不需要保存反向传播所需要的中间变量，速度更快
        correct_sum = 0  # 预测正确的总数
        for img, label in test_dataloader:
            img = img.to(device)
            label = label.to(device)
            label_predict = net(img)
            correct_sum += (label_predict.argmax(dim=1) == label).float().sum().item()

        acc_test = correct_sum / test_dataloader.__len__()
        print(f'epoch={epoch}, acc_test={acc_test}')




