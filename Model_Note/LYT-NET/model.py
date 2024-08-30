import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class LayerNormalization(nn.Module): # 实现层归一化操作，
    def __init__(self, dim):
        super(LayerNormalization, self).__init__() # LayerNorm 是归一化，但作用于每个样本而不是整个批次
        self.norm = nn.LayerNorm(dim) # dim是输入张量的通道数

    def forward(self, x): # 进行前向传播
        # Rearrange the tensor for LayerNorm (B, C, H, W) to (B, H, W, C)
        x = x.permute(0, 2, 3, 1) # permute调整张量的维度顺序，使其更适合layernorm操作
        x = self.norm(x) # 归一化
        # Rearrange back to (B, C, H, W)
        return x.permute(0, 3, 1, 2)

class SEBlock(nn.Module): # Squeeze and excitation 块，
    # 用于提升卷机神经网络CNN表现的模块
    # 主要作用是通过捕捉特征图的通道见依赖关系，动态地调整每个通道的权重，从而提升网络的表现。
    def __init__(self, input_channels, reduction_ratio=16):
        # input_channels输入特征图的通道数；
        # reduction_ratio 用于减少通道数的比率，默认是 16
        super(SEBlock, self).__init__() # 调用父类初始方法，
        
        # 自适应平均池化层
        # 将输入的每个通道的二维空间（高和宽）平均池化到一个单一的值
        # 无论输入特征图的空间尺寸是多少，输出的尺寸总是(C,1,1)，其中 C 是通道数
        self.pool = nn.AdaptiveAvgPool2d(1) 
        
        # 全连接层 fc1 & fc2
        # 将通道数从 input_channels 减少到 input_channels // reduction_ratio
        self.fc1 = nn.Linear(input_channels, input_channels // reduction_ratio) 
        # 将通道数从减少后的尺寸恢复到原始的 input_channels
        self.fc2 = nn.Linear(input_channels // reduction_ratio, input_channels)
        # reduction_ratio 主要用于减少模型的复杂度，通过减少通道数来减少计算量。
        self._init_weights()# 全中初始化方法，用于初始化全连接层的全中和偏置。
        
        # 权重初始化对于神经网络的训练非常重要，可以影响网络的收敛速度和最终性能。

    def forward(self, x): # 前向传播
        # x 是输入张量（通常是一个批次的特征图）。
        # batch_size 是输入批次的大小，num_channels 是输入特征图的通道数，
        # 后面两个下划线（_）表示忽略的维度（通常是高和宽）
        batch_size, num_channels, _, _ = x.size()
        
        # self.pool(x) 对输入 x 进行自适应平均池化将特征图的每个通道的空间维度池化到一个单一值。输出形状为 (batch_size, num_channels, 1, 1)
        # 然后，使用 .reshape(batch_size, num_channels) 将其重塑为 (batch_size, num_channels)，即将特征图的空间维度移除。
        y = self.pool(x).reshape(batch_size, num_channels)
        # 通过全连接层
        y = F.relu(self.fc1(y)) # 将池化后的特征图通过第一个全连接层，并应用ReLU激活函数，负值变成0，正值不变
        y = torch.tanh(self.fc2(y)) # 使用tanh将输出值压缩到-1到1之间。

        y = y.reshape(batch_size, num_channels, 1, 1) # 全连接层的输出重塑回(batch_size, num_channels, 1, 1)，可以与输入相乘
        # 通过元素级别的相乘，动态调整了每个通道的权重。
        # 这就是 SE 块的核心思想，通过学习到的权重增强重要的通道，抑制不重要的通道
        return x * y
    
    def _init_weights(self):
        # 初始化权重
        # kaiming_uniform_ 一种权重初始化方法，根据输入特征数的分布来初始化权重，适合使用ReLU激活函数的层。
        init.kaiming_uniform_(self.fc1.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.fc2.weight, a=0, mode='fan_in', nonlinearity='relu')
        # 用于将偏置初始化为常数（这里是 0）
        init.constant_(self.fc1.bias, 0)
        init.constant_(self.fc2.bias, 0)

class MSEFBlock(nn.Module): 
    # 初始化方法
    # filter 表示输入特征图的通道数
    def __init__(self, filters):
        super(MSEFBlock, self).__init__()
        # 层归一化
        self.layer_norm = LayerNormalization(filters)
        
        # 深度可分离卷积 Depthwise Convolution
        # kernel_size=3：卷积核的大小为 3x3。
        # padding=1：在卷积前对输入特征图的周围填充一圈像素，确保输出特征图的空间尺寸与输入一致
        # groups=filters：当 groups 参数等于 filters 时，
        # 这表示进行深度卷积，每个输入通道都独立进行卷积运算，而不跨通道进行卷积
        self.depthwise_conv = nn.Conv2d(filters, filters, kernel_size=3, padding=1, groups=filters)
        self.se_attn = SEBlock(filters)
        self._init_weights()

    def forward(self, x):
        x_norm = self.layer_norm(x)
        # x_norm 经过深度卷积层 depthwise_conv 后得到输出 x1。
        # 该操作处理了每个通道的局部空间信息。
        x1 = self.depthwise_conv(x_norm)
        # 通过SEBlock块
        x2 = self.se_attn(x_norm)

        x_fused = x1 * x2
        x_out = x_fused + x
        return x_out
    
    def _init_weights(self): # _init_weights 方法用于初始化深度卷积层的权重和偏置。
        # init.kaiming_uniform_ 使用 He 初始化方法，以均匀分布的方式初始化卷积层的权重，适合 ReLU 激活函数。
        init.kaiming_uniform_(self.depthwise_conv.weight, a=0, mode='fan_in', nonlinearity='relu')
        # init.constant_ 将卷积层的偏置初始化为常数（0）。
        init.constant_(self.depthwise_conv.bias, 0)

class MultiHeadSelfAttention(nn.Module): # 多头自注意力机制
    def __init__(self, embed_size, num_heads):
        super(MultiHeadSelfAttention, self).__init__() # 调用父类初始化方法，确保框架基本功能能正确初始化。
        self.embed_size = embed_size
        self.num_heads = num_heads

        # 检查嵌入大小是否可以被头数整除
        assert embed_size % num_heads == 0
        # 计算每个头的维度大小，每个头在嵌入维度中的分配大小。
        self.head_dim = embed_size // num_heads
        # 查询、键、值的线性变换
        self.query_dense = nn.Linear(embed_size, embed_size)
        self.key_dense = nn.Linear(embed_size, embed_size)
        self.value_dense = nn.Linear(embed_size, embed_size)
        # 定义头部的组合线性层
        # combine_heads 是将多个头的注意力输出组合在一起后的线性变换层，
        # 最终输出的嵌入维度与输入的 embed_size 保持一致。
        self.combine_heads = nn.Linear(embed_size, embed_size)
        # 初始化权重
        self._init_weights()

    def split_heads(self, x, batch_size): # 用于将输入张量x分割成多个头
        # 将张量x重新调整形状，以适应多个注意力头的计算。
        # batch_size 批次大小，
        # -1 自动计算出其他维度大小
        # num_heads 头数量
        # head_dim 每个头维度大小
        x = x.reshape(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        # batch_size 是批次大小，height 和 width 是输入特征图的高和宽，x.size() 获取输入张量的形状。
        batch_size, _, height, width = x.size()

        # 将输入张量展平为序列形式，以适应注意力机制的计算需求
        x = x.reshape(batch_size, height * width, -1)

        query = self.split_heads(self.query_dense(x), batch_size)
        key = self.split_heads(self.key_dense(x), batch_size)
        value = self.split_heads(self.value_dense(x), batch_size)
        # 计算注意力权重
        # torch.matmul(query, key.transpose(-2, -1))：计算查询和键的点积，得到注意力得分。
        # self.head_dim ** 0.5：对每个头的维度大小取平方根，用于缩放点积的结果
        # 最后用一个sofrmax进行归一化为概率分布
        attention_weights = F.softmax(torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5), dim=-1)
        
        # 使用注意力权重对值进行加权求和，得到注意力输出 attention。
        attention = torch.matmul(attention_weights, value)
        
        # 将 attention 张量的维度重新排列，将其转换回原始的形状。
        # contiguous()：确保张量在内存中是连续的，为后续操作做好准备。
        # reshape(batch_size, -1, self.embed_size)：将张量重新调整为 [batch_size, sequence_length, embed_size] 的形状。
        attention = attention.permute(0, 2, 1, 3).contiguous().reshape(batch_size, -1, self.embed_size)
        
        # 将多头注意力的输出 attention 通过线性层 combine_heads 进行投影，得到最终的输出 output。
        output = self.combine_heads(attention)
        
        # 将输出重新调整为 [batch_size, height, width, embed_size] 的形状
        # 让embed_size 作为第二个维度
        return output.reshape(batch_size, height, width, self.embed_size).permute(0, 3, 1, 2)

    def _init_weights(self):
        # 初始化所有权重
        init.xavier_uniform_(self.query_dense.weight)
        init.xavier_uniform_(self.key_dense.weight)
        init.xavier_uniform_(self.value_dense.weight)
        init.xavier_uniform_(self.combine_heads.weight)
        init.constant_(self.query_dense.bias, 0)
        init.constant_(self.key_dense.bias, 0)
        init.constant_(self.value_dense.bias, 0)
        init.constant_(self.combine_heads.bias, 0)

class Denoiser(nn.Module): # 去噪模块
    def __init__(self, num_filters, kernel_size=3, activation='relu'):
        # num_filters：卷积层中使用的过滤器数量，这决定了输出特征图的通道数。
        # kernel_size：卷积核的大小，默认值为 3。
        # activation：激活函数的名称，默认为 'relu'。
        super(Denoiser, self).__init__()

        # 第一个卷积层接收单通道输入（如灰度图像），
        # 输出 num_filters 个特征图，卷积核大小为 kernel_size，使用了边缘填充 padding=1。
        self.conv1 = nn.Conv2d(1, num_filters, kernel_size=kernel_size, padding=1)

        # 第二个卷积层将特征图下采样（stride=2），输出相同数量的特征图。 3次下采样
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, stride=2, padding=1)
        self.conv3 = nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, stride=2, padding=1)
        self.conv4 = nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, stride=2, padding=1)
        
        # 定义多头自注意力层
        self.bottleneck = MultiHeadSelfAttention(embed_size=num_filters, num_heads=4)

        # 定义三次上采样
        # scale_factor=2 指定了上采样的倍数，每个上采样层将输入的特征图放大 2 倍。
        # mode='nearest' 使用最近邻插值方法进行上采样。
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')

        # self.output_layer 是最终的输出层，将特征图转换为单通道输出。
        self.output_layer = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=1)

        # self.res_layer 是一个残差连接层，它将最后的上采样结果转换为单通道输出
        self.res_layer = nn.Conv2d(num_filters, 1, kernel_size=kernel_size, padding=1)
        # 获取并存储了指定的激活函数（默认为 relu），
        # 使用 getattr(F, activation) 来动态获取 torch.nn.functional 模块中的激活函数
        self.activation = getattr(F, activation) # activation = ReLu
        self._init_weights()

    def forward(self, x):
        x1 = self.activation(self.conv1(x))
        x2 = self.activation(self.conv2(x1))
        x3 = self.activation(self.conv3(x2))
        x4 = self.activation(self.conv4(x3))

        x = self.bottleneck(x4) # 多头注意力层，捕捉全局特征

        x = self.up4(x)
        x = self.up3(x + x3)
        x = self.up2(x + x2)
        x = x + x1

        ## ？？询问
        x = self.res_layer(x) # 通过残差层 res_layer 进行卷积，将特征图转换为单通道。
        return torch.tanh(self.output_layer(x + x)) # 通过输出层 output_layer，并应用 tanh 激活函数，输出最终的去噪结果
    
    def _init_weights(self): # 用于初始化模型中所有卷积层的权重和偏置
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.output_layer, self.res_layer]:
            init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='relu')
            # 使用 Kaiming 均匀初始化方法，以确保初始权重的分布适用于深度神经网络中的 ReLU 激活函数
            if layer.bias is not None:
                init.constant_(layer.bias, 0) # 将偏置初始化为常数 0。

class LYT(nn.Module):
    def __init__(self, filters=32): # filters表示卷积层中使用过滤器数量，32
        super(LYT, self).__init__() 
        # 定义了三个卷积处理层，分别为Y，CB，CR通道
        # 同时是一个辅助方法，用于创建一个卷积层和ReLU激活函数的组合
        self.process_y = self._create_processing_layers(filters)
        self.process_cb = self._create_processing_layers(filters)
        self.process_cr = self._create_processing_layers(filters)

        # 定义去噪器 粉笔去噪CB和CR通道
        self.denoiser_cb = Denoiser(filters // 2) # 这里//是什么意思？？
        self.denoiser_cr = Denoiser(filters // 2)

        # 定义亮度处理模块
        self.lum_pool = nn.MaxPool2d(8) # 是一个最大池化层，用于将 Y 通道的特征图下采样 8 倍
        # 是一个多头自注意力层，用于捕捉 Y 通道的全局信息。
        self.lum_mhsa = MultiHeadSelfAttention(embed_size=filters, num_heads=4) 
        # 是一个上采样层，将池化后的特征图放大 8 倍，以恢复原始尺寸。
        self.lum_up = nn.Upsample(scale_factor=8, mode='nearest') 
        # 是一个 1x1 的卷积层，用于对亮度特征图进行处理。
        self.lum_conv = nn.Conv2d(filters, filters, kernel_size=1, padding=0) 
        
        # 定义融合与调整模块
        # ref_conv一个1*1的卷积层，用于结合CB和CR通道的特征图
        self.ref_conv = nn.Conv2d(filters * 2, filters, kernel_size=1, padding=0)
        # 然后使用MSEFBlock模块，进一步处理融合后的特征图
        self.msef = MSEFBlock(filters)
        # recombine卷积层用于将亮度信息和参考的图像信息重新组合
        self.recombine = nn.Conv2d(filters * 2, filters, kernel_size=3, padding=1)
        # 最后的调整层，将融合后的特征图转化为RGB颜色空间中的三通道输出。
        self.final_adjustments = nn.Conv2d(filters, 3, kernel_size=3, padding=1)
        self._init_weights()

    # 处理层生成函数
    def _create_processing_layers(self, filters): 
        # 函数创建一个包含卷积层和ReLU激活层的顺序模块
        return nn.Sequential(
            nn.Conv2d(1, filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    # 定义RGB转YCbCr函数
    def _rgb_to_ycbcr(self, image):
        r, g, b = image[:, 0, :, :], image[:, 1, :, :], image[:, 2, :, :]
    
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.14713 * r - 0.28886 * g + 0.436 * b + 0.5
        v = 0.615 * r - 0.51499 * g - 0.10001 * b + 0.5
        
        yuv = torch.stack((y, u, v), dim=1) # stack函数将YUV通道堆叠在一起形成一个三通道的YCbCr图像
        return yuv

    def forward(self, inputs):
        # 将输入的RGB图像转换为YCbCr图像
        ycbcr = self._rgb_to_ycbcr(inputs)
        # 将YCbCr图像分割为三通道
        y, cb, cr = torch.split(ycbcr, 1, dim=1)

        # 分别对通道进行去噪处理，并于原始数据相加
        cb = self.denoiser_cb(cb) + cb
        cr = self.denoiser_cr(cr) + cr

        # 对每个通道分别进行卷积处理
        y_processed = self.process_y(y)
        cb_processed = self.process_cb(cb)
        cr_processed = self.process_cr(cr)

        ref = torch.cat([cb_processed, cr_processed], dim=1)
        lum = y_processed
        lum_1 = self.lum_pool(lum)
        lum_1 = self.lum_mhsa(lum_1)
        lum_1 = self.lum_up(lum_1)
        lum = lum + lum_1

        ref = self.ref_conv(ref)
        shortcut = ref
        ref = ref + 0.2 * self.lum_conv(lum)
        ref = self.msef(ref)
        ref = ref + shortcut

        recombined = self.recombine(torch.cat([ref, lum], dim=1))
        output = self.final_adjustments(recombined)
        return torch.sigmoid(output)
    
    def _init_weights(self):
        for module in self.children():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                init.kaiming_uniform_(module.weight, a=0, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    init.constant_(module.bias, 0)
                    
