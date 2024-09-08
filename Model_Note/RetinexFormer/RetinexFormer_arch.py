import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
from pdb import set_trace as stx
# import cv2


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


# input [bs,28,256,310]  output [bs, 28, 256, 256]
def shift_back(inputs, step=2):
    [bs, nC, row, col] = inputs.shape
    down_sample = 256 // row
    step = float(step) / float(down_sample * down_sample)
    out_col = row
    for i in range(nC):
        inputs[:, i, :, :out_col] = \
            inputs[:, i, :, int(step * i):int(step * i) + out_col]
    return inputs[:, :, :, :out_col]


# ----------------TAG1----------------
class Illumination_Estimator(nn.Module): # 继承nn.Module 这是pytorch中神经网络模块中的基类
    # 光照估计器

    def __init__(
            self, n_fea_middle, n_fea_in=4, n_fea_out=3):  #__init__部分是内部属性，而forward的输入才是外部输入
        # n_fea_middle 表示在中间层使用的特征图feature map和通道数 channels
        # n_fea_in=4 表示输入的特征图通道数，默认为4（包含RGB图像的3个通道和后面计算的平均值通道）
        # n_fea_out=3 表示输出的特征图通道数，默认为3（通常对应于RGB通道）
        
        super(Illumination_Estimator, self).__init__()
        # 调用了父类 nn.Module 的构造函数，确保父类的初始化方法被正确调用。
        # 这是Python类继承机制的一部分，super() 可以让我们调用父类的方法

        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)
        # 定义了一个二维卷积层conv1 用于提取图像的特征
        # n_fea_in：输入通道数（即输入特征图的深度）。
        # n_fea_middle：输出通道数（卷积后的特征图深度）。
        # kernel_size=1：卷积核的大小为1x1，这意味着每次卷积只在单个像素上操作，这通常用于调整通道数。
        # bias=True：是否添加偏置项（bias），偏置是一种额外的学习参数，用于帮助模型更好地拟合数据。

        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)
        # 定义另一个二维卷积层 depth_conv 这个卷积层使用了深度可分离卷积（depthwise convolution）
        # n_fea_middle：输入和输出的通道数相同（这个层不改变通道数）。
        # kernel_size=5：卷积核大小为5x5，这意味着每次卷积操作覆盖5x5的区域。
        # padding=2：填充，使得输出的空间维度与输入相同，padding=2 在每一侧添加2个像素的零填充。
        # groups=n_fea_in：这是深度可分离卷积的关键。每个输入通道都会有自己独立的一组卷积核，不同于标准卷积，它不会在通道间共享卷积核。


        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)
        # 定义的第三个卷积层 conv2
        # n_fea_middle：输入通道数。
        # n_fea_out：输出通道数（通常为3个通道的RGB图像）。
        # kernel_size=1：同样是1x1的卷积核，用于调整通道数


    def forward(self, img): # 函数定义了这个模块的前向传播过程，前向传播是指数数据通过网络的过程
                            # 在这个过程中，数据被不断转换为新的表达形式，直到输出
        # img:        b,c=3,h,w
        # mean_c:     b,c=1,h,w
        
        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w
        
        mean_c = img.mean(dim=1).unsqueeze(1) # 这一行计算了输入图像在通道维度上的均值
        # img.mean(dim=1)计算输入图像在通道维度（RGB）上的均值，生成一个形状为[batch_size, height, width]的特征图
        # unsqueeze(1)：在通道维度上添加一个新的维度，变为 [batch_size, 1, height, width]，这样就可以与原始输入图像在通道维度上进行拼接
        # stx()
        input = torch.cat([img,mean_c], dim=1) # 通道维度拼接 concatenate 生成一个新的输入特征图
        # 拼接后的特征图为 [batch_size, 4, height, width]
        
        # 将拼接后的特征图通过第一个卷积层 conv1，进行一次卷积操作。输出的特征图形状为 [batch_size, n_fea_middle, height, width]
        x_1 = self.conv1(input) 

        # 将经过 conv1 卷积后的特征图传入深度可分离卷积层 depth_conv，进一步提取特征。
        # 输出的特征图形状仍然为 [batch_size, n_fea_middle, height, width]
        illu_fea = self.depth_conv(x_1)

        # 将深度可分离卷积后的特征图传入最后一个卷积层 conv2，调整通道数，生成光照图（illumination map）。
        # 输出的光照图形状为 [batch_size, n_fea_out, height, width]
        illu_map = self.conv2(illu_fea)
        return illu_fea, illu_map


# ----------------TAG2----------------
class IG_MSA(nn.Module): # 模型中 illumination-Guided Multi-head Self-Attention
    # 类构造函数，用于初始化类的属性
    def __init__(
            self,
            dim, # 表示输入特征的维度（通常是通道数c）
            dim_head=64, # 每个注意力头的维度
            heads=8, # 注意力头的数量
    ):
        super().__init__()
        self.num_heads = heads # 注意力头的数量
        self.dim_head = dim_head # 每个注意力头的维度

        # 全连接层，to_q用于将输入特征映射到查询query向量空间
        # dim：输入的特征维度。
        # dim_head * heads：输出的特征维度，表示每个头的维度乘以头的数量。
        # bias=False：不添加偏置项
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        print(f"test11: {self.to_q.shape}")
        
        # 定义了一个可学习的参数rescale，用于缩放注意力权重，形状为[heads ,1,1]
        # 每个头都有一个独立的缩放参数
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))

        # 定义了一个全连接层 proj，用于将多头注意力输出的特征映射回原始维度
        # dim_head * heads：输入维度，注意力机制输出的维度。
        # dim：输出维度，恢复到输入特征的维度。
        # bias=True：添加偏置项
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)

        # 定义了一个位置编码（positional embedding）模块，用于对输入特征添加空间位置信息。
        # nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim)：一个深度可分离卷积，核大小为3x3，输入和输出维度相同。
        # groups=dim 表示每个通道独立卷积。
        # GELU()：激活函数，用于引入非线性。
        # nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim)：另一个深度可分离卷积，与前一个卷积层类似。  
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        # 保存输入特征的维度
        self.dim = dim

    def forward(self, x_in, illu_fea_trans):# 定义IG_MSA的前向传播过程
        """
        x_in: [b,h,w,c]         # input_feature
        illu_fea: [b,h,w,c]         # mask shift? 为什么是 b, h, w, c?
        return out: [b,h,w,c]
        """
        # 获取输入特征图的形状信息，分别是批次大小b，高度h，宽度w和通道数c
        b, h, w, c = x_in.shape
        
        print(f"11111x_in shape: {x_in.shape}")
        # 11111x_in shape: torch.Size([1, 256, 256, 32])
        
        print(f"12222illu_fea_trans shape: {illu_fea_trans.shape}")
        # 12222illu_fea_trans shape: torch.Size([1, 256, 256, 32])
        
        x = x_in.reshape(b, h * w, c)# 将输入特征图重新排列，hw表示高度和宽度的乘积
        q_inp = self.to_q(x)  
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        
        print(f"1333333q_inp shape: {q_inp.shape}")
        print(f"1444444k_inp shape: {k_inp.shape}")
        print(f"1555555v_inp shape: {v_inp.shape}")
        # shape: torch.Size([1, 65536, 32])
        
        
        # 传入光照特征
        illu_attn = illu_fea_trans # illu_fea: b,c,h,w -> b,h,w,c
        # 使用map和rearrange函数，进行形状变换
        q, k, v, illu_attn = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                 (q_inp, k_inp, v_inp, illu_attn.flatten(1, 2)))
        print(f"1666666q shape after rearrange: {q.shape}")
        print(f"17777777k shape after rearrange: {k.shape}")
        print(f"18888888v shape after rearrange: {v.shape}")
        print(f"199999999illu_attn shape after rearrange: {illu_attn.shape}")
        # 1666666q shape after rearrange: torch.Size([1, 1, 65536, 32])
        # 17777777k shape after rearrange: torch.Size([1, 1, 65536, 32])
        # 18888888v shape after rearrange: torch.Size([1, 1, 65536, 32])
        # 199999999illu_attn shape after rearrange: torch.Size([1, 1, 65536, 32])
        
        
        # 将值向量 v 与光照注意力 illu_attn 相乘，实现一种基于光照信息的加权机制。
        v = v * illu_attn
        print(f"20000000v shape after element-wise multiplication: {v.shape}")
        # 20000000v shape after element-wise multiplication: torch.Size([1, 1, 65536, 32])

        # q: b,heads,hw,c
        # 转置查询和键向量，使其形状变为 [b, heads, dim_head, hw]，为后续的矩阵乘法准备。
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)

        # 计算注意力得分 attn，即键和查询向量的点积，形状：b, heads, dim_head, dim_head
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        # 将注意力得分与可学习参数 rescale 相乘，进行缩放。
        attn = attn * self.rescale
        # 对注意力得分进行 softmax 操作，使其归一化到 [0, 1] 范围内。
        attn = attn.softmax(dim=-1)

        # 将注意力得分与值向量相乘，得到最终的加权特征图。
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose，对特征图
        x = x.reshape(b, h * w, self.num_heads * self.dim_head) # 将特征图重新排列

        out_c = self.proj(x).view(b, h, w, c) # 将特征图通过proj层，映射回原始维度 b,h,w,c
        
        # 将位置编码 out_p 与输出特征图 out_c 相加，得到最终的输出。
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(
            0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out

# ----------------TAG6----------------
class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1,
                      bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2).contiguous())
        return out.permute(0, 2, 3, 1)

# ----------------TAG3-----------------
class IGAB(nn.Module): # Illumination-Guided Attention Block
    def __init__(            
            self,
            dim,# 输入特征的维度（通常是通道数 c）。
            dim_head=64,# 每个注意力头的维度。
            heads=8,# 注意力头的数量。
            num_blocks=2,# 网络中块的数量，也就是 IG_MSA 和 FeedForward 组合的重复次数。
    ):
        super().__init__() # 调用父类nn.module的构建函数，确保正确初始化父类部分

        # 定义一个 ModuleList 来存储多个块。
        # ModuleList 是 PyTorch 提供的一个特殊列表，它可以存储多个子模块，并且可以像普通列表一样进行操作
        self.blocks = nn.ModuleList([])

        # 循环num_blocks，为每个块创建IG_MSA和PreNorm模块
        # PreNorm负责将输入特征进行归一化操作，然后传递给FeedForward层进行进一步处理
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                IG_MSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim))
            ]))
            # 一些讲解，这是一个循环，num_blocks决定了循环次数，也就是有多少个子模块会被添加到self.blocks这个列表中
            # self.blocks是一个nn.ModuleList，类似于python中的普通列表用于储存nn.Module子模块。
            # self.blocks.append(nn.ModuleList([ ... ])) 在每次循环中都会添加一个nn.ModuleList，包括两个子模块
            
            # IG_MSA(dim=dim, dim_head=dim_head, heads=heads)
            # IG_MSA 是之前看到的一个自定义类，用于实现多头自注意力机制。参数 dim 是输入特征的维度，dim_head 是每个头的维度大小，heads 是头的数量。
            # 在 self.blocks 中，attn（即注意力模块）被实例化为 IG_MSA。

            # PreNorm(dim, FeedForward(dim=dim))
            # PreNorm 是一个将归一化层应用在前馈神经网络之前的模块。这有助于模型在训练时稳定梯度。
            # FeedForward(dim=dim) 是一个标准的前馈神经网络模块，其输入输出维度都是 dim。


    def forward(self, x, illu_fea): # 定义了前向传播的过程
        """
        x: [b,c,h,w]
        illu_fea: [b,c,h,w]
        return out: [b,c,h,w]
        """
        # X是input
        # illu_fea是光照特征图
        print(f"x input shape (before permute): {x.shape}")
        # x input shape (before permute): torch.Size([1, 32, 256, 256])
        
        print(f"illu_fea input shape (before permute): {illu_fea.shape}")
        # illu_fea input shape (before permute): torch.Size([1, 32, 256, 256])
        
        
        x = x.permute(0, 2, 3, 1)# 对输入特征图 x 进行维度转换，形状变为 [b, h, w, c]

        for (attn, ff) in self.blocks:# 遍历每个块， attn是IG_MSA模块，ff是PreNorm模块
            
            # 将输入特征图 x 和光照特征 illu_fea 传递给 IG_MSA 层，计算注意力机制，
            # 并将结果与输入特征图相加（残差连接）
            x = attn(x, illu_fea_trans=illu_fea.permute(0, 2, 3, 1)) + x
            print(f"2111111x shape after attention: {x.shape}")
            # 2111111x shape after attention: torch.Size([1, 256, 256, 32])
            
            # 将 x 传递给 PreNorm 层，进行前馈神经网络处理，并将结果与输入特征图相加（残差连接）。
            x = ff(x) + x
            print(f"22222222x shape after feed-forward: {x.shape}")
            # 22222222x shape after feed-forward: torch.Size([1, 256, 256, 32])
        out = x.permute(0, 3, 1, 2)
        print(f"2333333333333out shape (after permute back): {out.shape}")
        # 2333333333333out shape (after permute back): torch.Size([1, 32, 256, 256])
        return out

# ----------------TAG4-----------------
class Denoiser(nn.Module):
    def __init__(self, in_dim=3, out_dim=3, dim=31, level=2, num_blocks=[2, 4, 4]):
        super(Denoiser, self).__init__()
        self.dim = dim
        self.level = level

        # Input projection
        # F0？这是一个卷积层，用于将输入特征投影到一个高维空间，初步提取特征。
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        # 编码器层: 由多个 IGAB 模块和卷积层组成。
        # IGAB 模块负责处理特征，同时卷积层用作下采样层，减少特征的分辨率并增加通道数。
        self.encoder_layers = nn.ModuleList([])
        dim_level = dim
        for i in range(level):
            self.encoder_layers.append(nn.ModuleList([
                IGAB(
                    dim=dim_level, num_blocks=num_blocks[i], dim_head=dim, heads=dim_level // dim),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False)
            ]))
            dim_level *= 2

        # Bottleneck F2？
        # 瓶颈层位于编码器和解码器之间，处理高维、低分辨率的特征。这个层的作用是捕捉全局信息，进一步提炼特征。
        self.bottleneck = IGAB(
            dim=dim_level, dim_head=dim, heads=dim_level // dim, num_blocks=num_blocks[-1])

        # Decoder
        # 解码器层: 使用反卷积层（ConvTranspose2d）将特征分辨率放大，并使用 IGAB 模块进行细化。
        # 每一层的 dim_level 会逐渐减小，直至回到原始输入的通道数。
        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_level, dim_level // 2, stride=2,
                                   kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
                IGAB(
                    dim=dim_level // 2, num_blocks=num_blocks[level - 1 - i], dim_head=dim,
                    heads=(dim_level // 2) // dim),
            ]))
            dim_level //= 2

        # Output projection
        # 输出投影层将处理后的特征映射回输出通道数，得到最终去噪后的结果。
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        # activation function
        # 使用 LeakyReLU 作为激活函数，并应用权重初始化函数 _init_weights 来初始化模型参数。
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, illu_fea): 
        """
        x:          [b,c,h,w]         x是feature, 不是image
        illu_fea:   [b,c,h,w]         illu_fea: 光照特征，用于辅助去噪任务。
        return out: [b,c,h,w]
        """

        # Embedding
        # 
        fea = self.embedding(x)

        # Encoder
        fea_encoder = []
        illu_fea_list = []
        for (IGAB, FeaDownSample, IlluFeaDownsample) in self.encoder_layers:
            fea = IGAB(fea,illu_fea)  # bchw
            illu_fea_list.append(illu_fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)
            illu_fea = IlluFeaDownsample(illu_fea)

        # Bottleneck
        fea = self.bottleneck(fea,illu_fea)

        # ----------------TAG45-----------------
        # Decoder
        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(
                torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))
            illu_fea = illu_fea_list[self.level-1-i]
            fea = LeWinBlcok(fea,illu_fea)

        # Mapping
        out = self.mapping(fea) + x

        return out

# 负责模型的单一阶段处理，主要由光照估计器（Illumination_Estimator）和去噪器（Denoiser）组成
class RetinexFormer_Single_Stage(nn.Module):
    # in_channels 和 out_channels: 输入和输出图像的通道数，通常为 3（RGB 图像）
    # # n_feat: 特征维度，决定网络中的通道数
    # level 和 num_blocks: 传递给去噪器，用于控制编码器-解码器的深度和每层的 IGAB 模块数量。
    def __init__(self, in_channels=3, out_channels=3, n_feat=31, level=2, num_blocks=[1, 1, 1]):
        
        super(RetinexFormer_Single_Stage, self).__init__()
        self.estimator = Illumination_Estimator(n_feat)
        self.denoiser = Denoiser(in_dim=in_channels,out_dim=out_channels,dim=n_feat,level=level,num_blocks=num_blocks)  #### 将 Denoiser 改为 img2img
    
    def forward(self, img):
        # img:        b,c=3,h,w
        
        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w

        illu_fea, illu_map = self.estimator(img)
        input_img = img * illu_map + img
        output_img = self.denoiser(input_img,illu_fea)

        return output_img


class RetinexFormer(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=31, stage=3, num_blocks=[1,1,1]):
        super(RetinexFormer, self).__init__()
        self.stage = stage

        modules_body = [RetinexFormer_Single_Stage(in_channels=in_channels, out_channels=out_channels, n_feat=n_feat, level=2, num_blocks=num_blocks)
                        for _ in range(stage)]
        
        self.body = nn.Sequential(*modules_body)
    
    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        out = self.body(x)

        return out


if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis
    model = RetinexFormer(stage=1,n_feat=32,num_blocks=[1,1,1]).cuda()
    print(model)
    inputs = torch.randn((1, 3, 256, 256)).cuda()
    flops = FlopCountAnalysis(model,inputs)
    n_param = sum([p.nelement() for p in model.parameters()])  # 所有参数数量
    print(f'GMac:{flops.total()/(1024*1024*1024)}')
    print(f'Params:{n_param}')