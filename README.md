# 教程
我们选取了几种具有代表性的方法，简要介绍将机器学习应用于超大规模集成电路（VLSI）物理设计周期的方法，为用户提供对`CircuirNet`功能和实用性的直观认识。有关整个示例，请参阅我们的 github 存储库[https://github.com/circuitnet/CircuitNet](https://github.com/circuitnet/CircuitNet)

请注意：所有三种方法都利用类图像特征来训练生成模型，例如全卷积网络 (FCN) 和U-Net网络，并将预测任务制定为图像到图像的转换任务。我们尽力重现了原始论文中的实验环境，包括模型架构、特征选择和损失。为了避免混淆，这些特征的名称和CircuitNet中的一致。

## 绕线拥塞预测
拥塞（congestion）是指在后端设计的绕线阶段中，由于需求过溢导致可用绕线资源不够的现象，常常被用作评估可布线性的指标，即基于当前设计解决方案的预期布线质量。拥塞预测对于指导布局阶段优化和减少总周转时间十分必要。

`Global Placement with Deep Learning-Enabled Explicit Routability Optimization`的网络[1]基于 FCN 的编码器-解码器架构将类图像特征转换为拥塞图。该架构如图 1 所示。

<div align='center'><img src="https://circuitnet.github.io/pics/tutorial/congestion_model.png" width = "600"></div>
<center><b>图 1</b> 模型架构</center>

该生成网络由编码器和解码器这两个基本模块组成，并根据图 1 所示的架构设计。

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

在这个过程中，我们选取了三个特征输入到模型中，包括 (1)macro_region、(2)RUDY、(3)RUDY_pin，它们经过预处理并通过提供的脚本`generate_training_set.py`组合成一个 numpy 数组（查看[快速启动页](https://circuitnet.github.io/intro/quickstart.html)以了解脚本的用法）。数组的可视化如图 2 所示。

<div align='center'><img src="https://circuitnet.github.io/pics/tutorial/congestion_input.png" width = "250" /></div>
<center><b>图 2</b> numpy数组的可视化</center>

我们创建一个名为`CongestionDataset`的类，负责接收拥塞特征和标签的 numpy 数组，同时通过 pytorch `DataLoader`进行读取和处理。

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

我们以端到端的方式训练这个网络，并计算输出和黄金拥塞图之间的损失，即CircuitNet中名为congestion_GR_horizontal_overflow 和congestion_GR_vertical_overflow的特征。

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
该模型通过200k iterations迭代训练模型。损失与训练迭代的曲线图如图 3 所示。归一化均方根误差 (NRMSE) 和结构相似性 (SSIM) 用于评估像素级精度，最终结果分别为 0.04 和 0.80。

<div align='center'><img src="https://circuitnet.github.io/pics/tutorial/congestion_loss.png"width = "250" ></div>
<center><b>图 3</b> 不同训练迭代的损失曲线图</center>

完成训练过程后，我们转储预测拥塞图的可视化，如图 4 所示。高对比度的部分表示拥塞热点。

<div align='center'><img src="https://circuitnet.github.io/pics/tutorial/congestion_output.png"width = "250" ></div>
<center><b>图 4</b> 拥塞预测图的可视化</center>

## 设计规则违反预测
设计规则检查 (DRC) 违反是对可布线性的另外一种估计方法。在全局布线后会出现拥塞，在详细布线后则出现DRC违反报错，但这在先进的技术节点上会有偏差，比如7nm制程工艺。因此，直接预测DRC违反也是必要的。`RouteNet: Routability Prediction for Mixed-Size Designs Using Convolutional Neural Network`[2]是准确预测违规热点的典型方法。该架构如图 5 所示。

<div align='center'><img src="https://circuitnet.github.io/pics/tutorial/DRC_model.png" width = "600"></div>
<center><b>图 5</b> 模型架构</center>

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

在这个过程中，我们选取了九个特征输入到模型中，包括 (1)macro_region，(2)cell_density，(3)RUDY_long，(4)RUDY_short，(5)RUDY_pin_long，(6)congestion_eGR_horizo​​ntal_overflow，(7)congestion_eGR_vertical_overflow，(8)congestion_GR_horizo​​ntal_overflow，(9)congestion_GR_vertical_overflow。这些特征同样经过预处理并组合成一个 numpy 数组。该数组的可视化如图 6 所示。

<div align='center'><img src="https://circuitnet.github.io/pics/tutorial/DRC_input.png"width = "250" ></div>
<center><b>图 6</b> 输入numpy数组的可视化</center>

我们创建一个名为`DRCDataset`的类，负责接收拥塞特征和标签的 numpy 数组，同时通过 pytorch `DataLoader`进行读取和处理。

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

我们以端到端的方式训练这个网络，并计算输出和黄金拥塞图之间的损失，即CircuitNet中名为DRC_all的特征。

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

该模型通过200k iterations迭代训练模型。训练损失曲线图如图3所示。

<div align='center'><img src="https://circuitnet.github.io/pics/tutorial/DRC_loss.png"width = "250" ></div>
<center><b>图 7</b> 不同训练迭代的损失曲线图</center>

DRC违反图显示了每个分块中的DRC违反数，即布局中每个Gcell中的DRC违反数。DRC 违反图的可视化如图 8 所示。

<div align='center'><img src="https://circuitnet.github.io/pics/tutorial/DRC_output.png"width = "250" ></div>
<center><b>图 8</b> DRC违反图可视化</center>

在这个过程中，违反次数超过阈值的分块称为热点，数量比非热点少得多，这种不平衡导致评估指标和接受者操作特征（ROC）曲线图的出现，用来评估该方法的性能。结果如图 9 所示。

<div align='center'><img src="https://circuitnet.github.io/pics/tutorial/drc_roc_routenet.png"width = "250" ></div>
<center><b>图9</b> ROC曲线</center>


## 电压降预测

IR drop是指电压偏离参考电压（VDD、VSS）的差异，必须进行限制，以避免时序和功能的退化。`AVIREC: ML-Aided Vectored IR-Drop Estimation and Classification` [3] 利用基于U-Net的网络来预测电压降。由于在时空轴上需要联合感知，MAVIREC引入了 3D 编码器来聚合时空特征，将预测结果输出为2D电压降图。

<div align='center'><img src="https://circuitnet.github.io/pics/tutorial/IR_model.png" width = "600"></div>
<center><b>图 10</b> 模型架构</center>

该生成网络由编码器和解码器这两个基本模块组成，模块的设计是根据图10所示的架构实现的。

```py
class DoubleConv3d(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DoubleConv2d(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)



class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
            DoubleConv3d(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv2d(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv2d(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
```

在这个过程中，我们选取了五个特征输入到模型中，包括(1)power_i, (2)power_s, (3)power_sca, (4)power_all, (5)power_t. Again。这些特征同样经过预处理并组合在一起作为一个 numpy 数组。该数组的可视化如图11所示。

<div align='center'><img src="https://circuitnet.github.io/pics/tutorial/IR_input.png"width = "250" ></div>
<center><b>图 11</b> 输入数组可视化</center>

我们创建一个名为`IRDropDataset`的类，负责接收拥塞特征和标签的 numpy 数组，同时通过pytorch `DataLoader`进行读取和处理。

```py
class IRDropDataset(object):
    def __init__(self, ann_file, dataroot, test_mode=False, **kwargs):
        super().__init__()
        self.ann_file = ann_file
        self.dataroot = dataroot
        self.test_mode = test_mode
        self.data_infos = self.load_annotations()

        self.temporal_key = 'Power_t'

    def load_annotations(self):  
        data_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                infos = line.strip().split(',')
                label = infos[-1]
                features = infos[:-1]
                info_dict = dict()
                if self.dataroot is not None:
                    for feature in features:
                        info_dict[feature.split('/')[0]] = osp.join(self.dataroot, feature)
                    feature_path = info_dict
                    label_path = osp.join(self.dataroot, label)
                data_infos.append(dict(feature_path=feature_path, label_path=label_path))
        return data_infos

    def prepare_data(self, idx):
        results = copy.deepcopy(self.data_infos[idx])

        feature = np.load(results['feature_path']).transpose(2, 0, 1).astype(np.float32)
        feature = np.expand_dims(feature, axis=0)
        label = np.load(results['label_path']).transpose(2, 0, 1).astype(np.float32).squeeze()
        return feature, label, results['label_path']


    def __len__(self):
        return len(self.data_infos)


    def __getitem__(self, idx):
        return self.prepare_data(idx)
```

我们以端到端的方式训练这个网络，并计算输出和黄金拥塞图之间的损失，即CircuitNet中名为ir_drop的特征。

```py
class MAVI(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 bilinear=False,
                 init_cfg=dict(type='normal', gain=0.02), 
                 **kwargs):
        super(MAVI, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.inc = DoubleConv3d(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1

        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.outc = OutConv(64, out_channels)

        self.init_type = 'normal' if init_cfg is None else init_cfg.get(
            'type', 'normal')
        self.init_gain = 0.02 if init_cfg is None else init_cfg.get(
            'gain', 0.02)

    def forward(self, x):
        x_in = x[:, :, :self.out_channels, :, :] # [b c 4 h w]
        x1 = self.inc(x)
        x2 = self.down1(x1)  # [1, 64, 20, 256, 256]
        x3 = self.down2(x2)  # [1, 128, 16, 128, 128]
        x4 = self.down3(x3)  # [1, 512, 12, 64, 64]

        x = self.up1(x4.mean(dim=2), x3.mean(dim=2))
        x = self.up2(x, x2.mean(dim=2))
        x = self.up3(x, x1.mean(dim=2))
        logits = self.outc(x)

        logits = x_in.squeeze(1)*logits
        return torch.sum(logits, dim=1)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            load_checkpoint(self, pretrained, strict=False, logger=None)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    constant_init(m.weight, 1)
                    constant_init(m.bias, 0)

                if isinstance(m, nn.Conv3d):
                    kaiming_init(m)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)
        else:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')
```

电压降图显示了每个分块中最大的电压降值，即布局中每个Gcell中的最大电压降值。电压降图可视化如图 12 所示。

<div align='center'><img src="https://circuitnet.github.io/pics/tutorial/IR_output.png"width = "250" ></div>
<center><b>图 12</b> 电压降图可视化</center>

该模型通过200k iterations迭代训练模型。训练损失曲线图如图13 所示。

<div align='center'><img src="https://circuitnet.github.io/pics/tutorial/IR_loss.png"width = "250" ></div>
<center><b>图 13</b> 不同训练迭代的损失曲线图</center>

在这个过程中，电压降值超过阈值的分块称为热点，因此，我们采用与DRC违反预测任务相同的评估指标，即接收者操作特征（ROC）曲线，来评估该方法的性能。结果如图14所示。

<div align='center'><img src="https://circuitnet.github.io/pics/tutorial/irdrop_roc_mavi.png"width = "250" ></div>
<center><b>图 14</b> ROC曲线图</center>

## 引文
```
[1] S. Liu, et al. “Global Placement with Deep Learning- Enabled Explicit Routability Optimization,” in DATE 2021. 1821–1824.

[2] Z. Xie, et al. “RouteNet: Routability prediction for mixed-size designs using convolutional neural network,” in ICCAD 2018. 1–8.

[3] V. A. Chhabria, et al. “MAVIREC: ML-Aided Vectored IR-Drop Estimation and Classification,” in DATE 2021. 1825–1828.
```
