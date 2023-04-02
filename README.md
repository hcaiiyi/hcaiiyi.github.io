# 教程
在此，我们选择几种具有代表性的方法来简要介绍将机器学习应用于 VLSI 物理设计周期，为`CircuirNet`用户提供对功能和实用性的直观认识。有关整个示例，请参阅我们的 github 存储库[https://github.com/circuitnet/CircuitNet](https://github.com/circuitnet/CircuitNet)。

请注意，所有三种选定的方法都利用类似图像的特征来训练生成模型，例如完全卷积网络 (FCN) 和 U-Net，将预测任务制定为图像到图像的转换任务。我们尽力重现了原始论文中的实验环境，包括模型架构、特征选择和损失。特征的名称与 CircuitNet 中的名称相匹配，以避免混淆。
## 拥塞预测
拥塞定义为在后端设计的布线阶段，布线需求超过可用布线资源的溢出。它经常被用作评估可布线性的指标，即基于当前设计解决方案的布线的预期质量。拥塞预测对于指导布局阶段的优化和减少总周转时间是必要的。

[1]的网络`Global Placement with Deep Learning-Enabled Explicit Routability Optimization`使用基于 FCN 的编码器-解码器架构将类图像特征转换为拥塞图。该架构如**图 1** 所示。
<div align='center'><img src="https://circuitnet.github.io/pics/tutorial/congestion_model.png"></div>
<center>**图1** 模型架构</center>

生成网络由两个基本模块组成，编码器和解码器，它们是根据图 1 所示的架构设计的。
```py
class conv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True):
        super(conv, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        return self.main(input)

class upconv(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(upconv, self).__init__()
        self.main = nn.Sequential(
                nn.ConvTranspose2d(dim_in, dim_out, 4, 2, 1),
                nn.InstanceNorm2d(dim_out, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)

class Encoder(nn.Module):
    def __init__(self, in_dim=3, out_dim=32):
        super(Encoder, self).__init__()
        self.in_dim = in_dim
        self.c1 = conv(in_dim, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.c2 = conv(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.c3 = nn.Sequential(
                nn.Conv2d(64, out_dim, 3, 1, 1),
                nn.BatchNorm2d(out_dim),
                nn.Tanh()
                )

    def init_weights(self):
        generation_init_weights(self)


    def forward(self, input):
        h1 = self.c1(input)
        h2 = self.pool1(h1)
        h3 = self.c2(h2)
        h4 = self.pool2(h3)
        h5 = self.c3(h4)
        return h5, h2  # shortpath from 2->7


class Decoder(nn.Module):
    def __init__(self, out_dim=2, in_dim=32):
        super(Decoder, self).__init__()
        self.conv1 = conv(in_dim, 32)
        self.upc1 = upconv(32, 16)
        self.conv2 = conv(16, 16)
        self.upc2 = upconv(32+16, 4)
        self.conv3 =  nn.Sequential(
                nn.Conv2d(4, out_dim, 3, 1, 1),
                nn.Sigmoid()
                )

    def init_weights(self):
        generation_init_weights(self)

    def forward(self, input):
        feature, skip = input
        d1 = self.conv1(feature)
        d2 = self.upc1(d1)
        d3 = self.conv2(d2)
        d4 = self.upc2(torch.cat([d3, skip], dim=1))
        output = self.conv3(d4)  # shortpath from 2->7
        return output  return self.main(input)
```
