import enum
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
dataloader = torch.utils.data.DataLoader(dataset=mnist, batch_size = batch_size, shuffle = True)

#判别网络
class discriminator(nn.Module):
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
class generator(nn.Module):
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
    for i, (img, _) in enumerate(dataloader):
        num_img = img.size(0)
        #**********************训练判别器
        img = img.view(num_img, -1)#将图片展开28*28
        real_img = Variable(img)
        real_label = Variable(torch.ones(num_img))#定义真实label为1
        fake_label = Variable(torch.zeros(num_img))#定义假label为0
        #print(fake_label.shape)

        #计算real_img 的损失
        real_out = D(real_img)#将真实图片放入判别器
        real_out = real_out.squeeze()#去掉维度为1的 torch.size()
        #print(real_out.shape)
        d_loss_real = criterion(real_out, real_label)#得到真实图片的Loss
        real_scores = real_out #越接近越好

        #计算fake_img 的损失
        z = Variable(torch.randn(num_img, z_dimension))#随机生成一些噪声
        fake_img = G(z) #放入生成网络生成一张假的照片
        fake_out = D(fake_img)#判别器判断假的图片
        fake_out = fake_out.squeeze()
        d_loss_fake = criterion(fake_out, fake_label)
        fake_scores = fake_out#越接近越好   

        #反向传播和优化
        d_loss = d_loss_fake + d_loss_real #将真假图片loss加起来
        d_optimizer.zero_grad()#每次梯度归零
        d_loss.backward()#反向传播
        d_optimizer.step()#更新参数

        #****************************训练生成器
        #计算fake_img 的损失
        #z = Variable(torch.randn(num_img, z_dimension)).cuda() #得到随机噪声
        z = Variable(torch.randn(num_img, z_dimension)) #得到随机噪声
        fake_img = G(z)#生成假的图片
        output = D(fake_img)#经过判别器得到的结果
        output = output.squeeze()
        g_loss = criterion(output, real_label)#得到假的图片和真的图片的loss

        #反向传播和优化
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if(i+1)%100 == 0:
            print('epoch %d[{}/{}], d_loss:{:,.6f}, g_loss:{:,.6f}, D real:{:,.6f}, D fake:{:,.6f}'.format(epoch, num_epoch, d_loss.item(), g_loss.item(),real_scores.data.mean(), fake_scores.data.mean()))

    if epoch ==0:
        real_images = to_img(real_img.cpu().data)
        save_image(real_images, 'real_images.png')
    
    fake_images = to_img(fake_img.cpu().data)
    save_image(fake_images, 'fake_imgs--{}.png'.format(epoch+1))

torch.save(G.state_dict(), 'generator.pth')
torch.save(D.state_dict(),'discriminator.pth')

