from logging import critical
from random import shuffle
from turtle import forward
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import os

if not os.path.exists('img'):
    os.mkdir('img')

def to_img(x):
    out = 0.5*(x+1)
    out = out.clamp(0, 1)
    out = out.view(-1, 1, 28, 28)
    return out

#初始化参数
batch_size = 128
num_epoch = 10
z_dimension = 100

img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.13707,), (0.3081))])

mnist = datasets.MNIST(
    root = '', train=True, transform=img_transform, download=True
)
#数据集加载
dataloader = torch.utils.data.DataLoader(datasets=mnist, batch_size = batch_size, shuffle = True)

#判别网络
class discriminator(nn.Model):
    def __init__(self) -> None:
        super(discriminator, self).__init__()
        self.dis = nn.Sequential(nn.Linear(784, 256),
                                nn.LeakyReLU(0.2), 
                                nn.Linear(256, 256),
                                nn.LeakyReLU(0.2),
                                nn.Linear(256, 1),
                                nn.Sigmoid())
        def forward(self, x):
            x = self.dis(x)
            return x


#生成器
class generator(nn.Model):
    def __init__(self) -> None:
        super(generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 784),
            nn.Tanh()#tanh 激活函数是希望生成的假的图片数据能够分布在-1到1之间
        )

    def forward(self,x):
        x = self.gen(x)
        return x


D = discriminator()
G = generator()

if torch.cuda.is_available():
    D = D.cuda()
    G = G.cuda()
#判别器的训练由两部分组成：第一部分是真的图像判别为真，第二部分是假的图片为假，在这两个过程当中，生成器的参数不参与更新
#二进制交叉熵损失和优化器

criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0003)
g_optimizer = torch.optim.Adam(G.parameters(), lr = 0.0003)

#开始训练
for epoch in range(num_epoch):
