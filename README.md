# 教程
在此，我们选择几种具有代表性的方法来简要介绍将机器学习应用于 VLSI 物理设计周期，为`CircuirNet`用户提供对功能和实用性的直观认识。有关整个示例，请参阅我们的 github 存储库[https://github.com/circuitnet/CircuitNet](https://github.com/circuitnet/CircuitNet)。

请注意，所有三种选定的方法都利用类似图像的特征来训练生成模型，例如完全卷积网络 (FCN) 和 U-Net，将预测任务制定为图像到图像的转换任务。我们尽力重现了原始论文中的实验环境，包括模型架构、特征选择和损失。特征的名称与 CircuitNet 中的名称相匹配，以避免混淆。
## 绕线拥塞预测
拥塞定义为在后端设计的布线阶段，布线需求超过可用布线资源的溢出。它经常被用作评估可布线性的指标，即基于当前设计解决方案的布线的预期质量。拥塞预测对于指导布局阶段的优化和减少总周转时间是必要的。

[1]的网络`Global Placement with Deep Learning-Enabled Explicit Routability Optimization`使用基于 FCN 的编码器-解码器架构将类图像特征转换为拥塞图。该架构如**图1** 所示。
<div align='center'><img src="https://circuitnet.github.io/pics/tutorial/congestion_model.png"></div>
<center>图1 模型架构</center>

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

在这项工作中，选择了三个特征作为输入特征以输入模型。包含的功能是 (1)macro_region、(2)RUDY、(3)RUDY_pin，它们经过预处理并通过提供的脚本组合在一起作为一个 numpy 数组（查看快速启动页面以了解脚本的`generate_training_set.py`用法）。阵列的可视化如图 2 所示。
<div align='center'><img src="https://circuitnet.github.io/pics/tutorial/congestion_input.png"></div>
<center>**图2** numpy数组的可视化</center>

我们创建一个名为的类`CongestionDataset`来获取拥塞特征和标签的 numpy 数组，同时通过 pytorch 读取和处理它们`DataLoader`。

```py
class CongestionDataset(object):
    def __init__(self, ann_file, dataroot, pipeline=None, test_mode=False, **kwargs):
        super().__init__()
        self.ann_file = ann_file
        self.dataroot = dataroot
        self.test_mode = test_mode
        if pipeline:
            self.pipeline = Compose(pipeline)
        else:
            self.pipeline = None

        self.data_infos = self.load_annotations()

    def load_annotations(self):
        data_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                feature, label = line.strip().split(',')
                if self.dataroot is not None:
                    feature_path = osp.join(self.dataroot, feature)
                    label_path = osp.join(self.dataroot, label)
                data_infos.append(dict(feature_path=feature_path, label_path=label_path))
        return data_infos

    def prepare_data(self, idx):
        results = copy.deepcopy(self.data_infos[idx])
        results['feature'] = np.load(results['feature_path'])
        results['label'] = np.load(results['label_path'])

        results = self.pipeline(results) if self.pipeline else results

        feature =  results['feature'].transpose(2, 0, 1).astype(np.float32)
        label = results['label'].transpose(2, 0, 1).astype(np.float32)

        return feature, label, results['label']

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        return self.prepare_data(idx)
```

我们以端到端的方式训练这个网络，并计算输出和黄金拥塞图之间的损失，这是来自 CircuitNet 的名为 congestion_GR_horizo​​ntal_overflow 和 congestion_GR_vertical_overflow 的特征。

```py
class GPDL(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=2,
                 **kwargs):
        super().__init__()

        self.encoder = Encoder(in_dim=in_channels)
        self.decoder = Decoder(out_dim=out_channels)

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)

    def init_weights(self, pretrained=None, pretrained_transfer=None, strict=False, **kwargs):
        if isinstance(pretrained, str):
            new_dict = OrderedDict()
            weight = torch.load(pretrained, map_location='cpu')['state_dict']
            for k in weight.keys():
                new_dict[k] = weight[k]
            load_state_dict(self, new_dict, strict=strict, logger=None)
        elif pretrained is None:
            generation_init_weights(
                self, init_type=self.init_type, init_gain=self.init_gain)
        else:
            raise TypeError("'pretrained' must be a str or None. "
                            f'But received {type(pretrained)}.')
```
该模型经过 200k 次迭代训练。损失与训练迭代的曲线如图 3 所示。 归一化均方根误差 (NRMSE) 和结构相似性指数度量 (SSIM) 用于评估像素级精度，这些指标的最终结果为 0.04 和分别为 0.80。

<div align='center'><img src="https://circuitnet.github.io/pics/tutorial/congestion_loss.png"></div>
<center>**图3** 不同训练迭代的训练损失</center>

完成训练过程后，我们转储预测拥塞图的可视化，如图 4 所示。高对比度的部分表示拥塞热点。

<div align='center'><img src="https://circuitnet.github.io/pics/tutorial/congestion_output.png"></div>
<center>**图4** 预测拥堵图的可视化</center>
## 设计规则违反预测
违反设计规则检查 (DRC) 是对可布线性的另一种估计。全局路由后出现拥塞，详细路由后上报DRC违规。并且在先进的技术节点上它们之间存在偏差，例如7nm。因此，也有必要直接预测 DRC 违规。`RouteNet: Routability Prediction for Mixed-Size Designs Using Convolutional Neural Network`[2]是准确预测违规热点的典型方法。该架构如图 5 所示。

<div align='center'><img src="https://circuitnet.github.io/pics/tutorial/DRC_model.png
"></div>
<center>**图5** 模型架构</center>

该网络与拥塞预测中的网络相同。

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

在这项工作中，九个特征被选为输入特征以输入模型。包含的特征是 (1)macro_region，(2)cell_density，(3)RUDY_long，(4)RUDY_short，(5)RUDY_pin_long，(6)congestion_eGR_horizo​​ntal_overflow，(7)congestion_eGR_vertical_overflow，(8)congestion_GR_horizo​​ntal_overflow，(9)congestion_GR_vertical_overflow。同样，这些特征被预处理并组合在一起作为一个 numpy 数组。阵列的可视化如图 6 所示。

<div align='center'><img src="https://circuitnet.github.io/pics/tutorial/DRC_input.png"></div>
<center>**图6** 输入numpy数组的可视化</center>

我们创建一个名为的类`DRCDataset`来获取拥塞特征和标签的 numpy 数组，同时通过 pytorch 读取和处理它们`DataLoader`。

```py
class DRCDataset(object):
    def __init__(self, ann_file, dataroot, test_mode=None, **kwargs):
        super().__init__()
        self.ann_file = ann_file
        self.dataroot = dataroot
        self.test_mode = test_mode
        self.data_infos = self.load_annotations()

    def load_annotations(self):
        data_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                feature, label = line.strip().split(',')
                if self.dataroot is not None:
                    feature_path = osp.join(self.dataroot, feature)
                    label_path = osp.join(self.dataroot, label)
                data_infos.append(dict(feature_path=feature_path, label_path=label_path))
        return data_infos

    def prepare_data(self, idx):
        results = copy.deepcopy(self.data_infos[idx])

        feature = np.load(results['feature_path']).transpose(2, 0, 1).astype(np.float32)
        label = np.load(results['label_path']).transpose(2, 0, 1).astype(np.float32)

        return feature, label, results['label_path']

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        return self.prepare_data(idx)
```

我们以端到端的方式训练这个网络，并计算输出和黄金 DRC 违规图之间的损失，这是来自 CircuitNet 的名为 DRC_all 的特征。

```py
class RouteNet(nn.Module):
    def __init__(self,
                 in_channels=9,
                 out_channels=2,
                 **kwargs):
        super().__init__()

        self.encoder = Encoder(in_dim=in_channels)
        self.decoder = Decoder(out_dim=out_channels)

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)

    def init_weights(self, pretrained=None, pretrained_transfer=None, strict=False, **kwargs):
        if isinstance(pretrained, str):
            new_dict = OrderedDict()
            weight = torch.load(pretrained, map_location='cpu')['state_dict']
            for k in weight.keys():
                new_dict[k] = weight[k]

            new_dict_clone = new_dict.copy()
            for key, value in new_dict_clone.items():
                if key.endswith(('running_mean', 'running_var')):
                    del new_dict[key]

            load_state_dict(self, new_dict, strict=strict, logger=None)
        elif pretrained is None:
            generation_init_weights(
                self, init_type=self.init_type, init_gain=self.init_gain)
        else:
            raise TypeError("'pretrained' must be a str or None. "
                            f'But received {type(pretrained)}.')
```

该模型经过 200k 次迭代训练。训练损失曲线如图 7 所示。

<div align='center'><img src="https://circuitnet.github.io/pics/tutorial/DRC_loss.png"></div>
<center>**图7** 不同训练迭代的训练损失</center>

DRC 违例图提供每个区块中的 DRC 违例数，即布局中每个 Gcell 中的违例数。DRC 违规地图的可视化如图 8 所示。

<div align='center'><img src="https://circuitnet.github.io/pics/tutorial/DRC_output.png"></div>
<center>**图8** 可视化</center>

在这项工作中，违规次数超过阈值的图块被视为热点。热点比非热点少得多，这是不平衡的，因此采用评估指标，接受者操作特征（ROC）曲线来评估该方法的性能。结果如图 9 所示。

<div align='center'><img src="https://circuitnet.github.io/pics/tutorial/drc_roc_routenet.png">**图9** ROC曲线</center>

## 电压降预测
