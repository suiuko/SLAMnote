import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.transforms.functional as TF

class Denoiser(nn.Module):
    def __init__(self, filters):
        super(Denoiser, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.conv(x)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_size = embed_size
        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        values = values.reshape(N, value_len, self.num_heads, self.embed_size // self.num_heads)
        keys = keys.reshape(N, key_len, self.num_heads, self.embed_size // self.num_heads)
        queries = queries.reshape(N, query_len, self.num_heads, self.embed_size // self.num_heads)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = torch.nn.functional.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.embed_size
        )

        out = self.fc_out(out)
        return out

class MSEFBlock(nn.Module):
    def __init__(self, filters):
        super(MSEFBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class LYT(nn.Module):
    def __init__(self, filters=32):
        super(LYT, self).__init__()
        self.process_l = self._create_processing_layers(filters)
        self.process_a = self._create_processing_layers(filters)
        self.process_b = self._create_processing_layers(filters)

        self.denoiser_a = Denoiser(filters // 2)
        self.denoiser_b = Denoiser(filters // 2)
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
    
    def _rgb_to_lab(self, image):
        # 使用 torchvision.transforms.functional 的方法将 RGB 转换为 LAB
        lab = TF.rgb_to_lab(image)
        # 分离 L, a, b 通道
        l, a, b = lab[:, 0:1, :, :], lab[:, 1:2, :, :], lab[:, 2:3, :, :]
        return l, a, b

    def forward(self, inputs):
        l, a, b = self._rgb_to_lab(inputs)
        a = self.denoiser_a(a) + a
        b = self.denoiser_b(b) + b

        l_processed = self.process_l(l)
        a_processed = self.process_a(a)
        b_processed = self.process_b(b)

        ref = torch.cat([a_processed, b_processed], dim=1)
        lum = l_processed
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

# 测试代码
if __name__ == "__main__":
    # 假设输入是一个批次的 RGB 图像，大小为 (batch_size, 3, height, width)
    inputs = torch.randn(1, 3, 256, 256)  # 示例输入
    model = LYT(filters=32)
    output = model(inputs)
    print(output.shape)  # 输出的形状
