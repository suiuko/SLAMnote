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
        y = F.relu(self.fc1(y))
        y = torch.tanh(self.fc2(y))
        y = y.reshape(batch_size, num_channels, 1, 1)
        return x * y
    
    def _init_weights(self):
        init.kaiming_uniform_(self.fc1.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.fc2.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.constant_(self.fc1.bias, 0)
        init.constant_(self.fc2.bias, 0)

class MSEFBlock(nn.Module):
    def __init__(self, filters):
        super(MSEFBlock, self).__init__()
        self.layer_norm = LayerNormalization(filters)
        self.depthwise_conv = nn.Conv2d(filters, filters, kernel_size=3, padding=1, groups=filters)
        self.se_attn = SEBlock(filters)
        self._init_weights()

    def forward(self, x):
        x_norm = self.layer_norm(x)
        x1 = self.depthwise_conv(x_norm)
        x2 = self.se_attn(x_norm)
        x_fused = x1 * x2
        x_out = x_fused + x
        return x_out
    
    def _init_weights(self):
        init.kaiming_uniform_(self.depthwise_conv.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.constant_(self.depthwise_conv.bias, 0)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        assert embed_size % num_heads == 0
        self.head_dim = embed_size // num_heads
        self.query_dense = nn.Linear(embed_size, embed_size)
        self.key_dense = nn.Linear(embed_size, embed_size)
        self.value_dense = nn.Linear(embed_size, embed_size)
        self.combine_heads = nn.Linear(embed_size, embed_size)
        self._init_weights()

    def split_heads(self, x, batch_size):
        x = x.reshape(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        x = x.reshape(batch_size, height * width, -1)

        query = self.split_heads(self.query_dense(x), batch_size)
        key = self.split_heads(self.key_dense(x), batch_size)
        value = self.split_heads(self.value_dense(x), batch_size)
        
        attention_weights = F.softmax(torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5), dim=-1)
        attention = torch.matmul(attention_weights, value)
        attention = attention.permute(0, 2, 1, 3).contiguous().reshape(batch_size, -1, self.embed_size)
        
        output = self.combine_heads(attention)
        
        return output.reshape(batch_size, height, width, self.embed_size).permute(0, 3, 1, 2)

    def _init_weights(self):
        init.xavier_uniform_(self.query_dense.weight)
        init.xavier_uniform_(self.key_dense.weight)
        init.xavier_uniform_(self.value_dense.weight)
        init.xavier_uniform_(self.combine_heads.weight)
        init.constant_(self.query_dense.bias, 0)
        init.constant_(self.key_dense.bias, 0)
        init.constant_(self.value_dense.bias, 0)
        init.constant_(self.combine_heads.bias, 0)

class Denoiser(nn.Module):
    def __init__(self, num_filters, kernel_size=3, activation='relu'):
        super(Denoiser, self).__init__()
        self.conv1 = nn.Conv2d(1, num_filters, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, stride=2, padding=1)
        self.conv3 = nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, stride=2, padding=1)
        self.conv4 = nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, stride=2, padding=1)
        self.bottleneck = MultiHeadSelfAttention(embed_size=num_filters, num_heads=4)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.output_layer = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=1)
        self.res_layer = nn.Conv2d(num_filters, 1, kernel_size=kernel_size, padding=1)
        self.activation = getattr(F, activation)
        self._init_weights()

    def forward(self, x):
        x1 = self.activation(self.conv1(x))
        x2 = self.activation(self.conv2(x1))
        x3 = self.activation(self.conv3(x2))
        x4 = self.activation(self.conv4(x3))
        x = self.bottleneck(x4)
        x = self.up4(x)
        x = self.up3(x + x3)
        x = self.up2(x + x2)
        x = x + x1
        x = self.res_layer(x)
        return torch.tanh(self.output_layer(x + x))
    
    def _init_weights(self):
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.output_layer, self.res_layer]:
            init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='relu')
            if layer.bias is not None:
                init.constant_(layer.bias, 0)

class LYT(nn.Module):
    def __init__(self, filters=32):
        super(LYT, self).__init__()
        self.process_y = self._create_processing_layers(filters)
        self.process_cb = self._create_processing_layers(filters)
        self.process_cr = self._create_processing_layers(filters)

        self.denoiser_cb = Denoiser(filters // 2)
        self.denoiser_cr = Denoiser(filters // 2)
        self.lum_pool = nn.MaxPool2d(8)
        self.lum_mhsa = MultiHeadSelfAttention(embed_size=filters, num_heads=4)
        self.lum_up = nn.Upsample(scale_factor=8, mode='nearest')
        self.lum_conv = nn.Conv2d(filters, filters, kernel_size=1, padding=0)
        self.ref_conv = nn.Conv2d(filters * 2, filters, kernel_size=1, padding=0)
        self.msef = MSEFBlock(filters)
        self.recombine = nn.Conv2d(filters * 2, filters, kernel_size=3, padding=1)
        self.final_adjustments = nn.Conv2d(filters, 3, kernel_size=3, padding=1)
        self._init_weights()

    def _create_processing_layers(self, filters):
        return nn.Sequential(
            nn.Conv2d(1, filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def _rgb_to_ycbcr(self, image):
        r, g, b = image[:, 0, :, :], image[:, 1, :, :], image[:, 2, :, :]
    
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.14713 * r - 0.28886 * g + 0.436 * b + 0.5
        v = 0.615 * r - 0.51499 * g - 0.10001 * b + 0.5
        
        yuv = torch.stack((y, u, v), dim=1)
        return yuv

    def forward(self, inputs):
        ycbcr = self._rgb_to_ycbcr(inputs)
        y, cb, cr = torch.split(ycbcr, 1, dim=1)
        cb = self.denoiser_cb(cb) + cb
        cr = self.denoiser_cr(cr) + cr

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
                    
