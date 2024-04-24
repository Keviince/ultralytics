import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ('BottleneckAttention', 'CustomBottleneck')

class BottleneckAttention(nn.Module):
    def __init__(self, dim, fmap_size, heads=4):
        super(BottleneckAttention, self).__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        
        # Positional embeddings
        self.rel_height = nn.Parameter(torch.randn(fmap_size[0] * 2 - 1, dim // heads))
        self.rel_width = nn.Parameter(torch.randn(fmap_size[1] * 2 - 1, dim // heads))

        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.attn_out = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: t.reshape(b, self.heads, c // self.heads, h * w), qkv)

        q *= self.scale
        k = k.permute(0, 1, 3, 2)

        content_content_attn = torch.matmul(q, k)
        
        # Add positional embeddings
        rel_height = self.rel_height[:h * 2 - 1, :].unsqueeze(0).unsqueeze(0)
        rel_width = self.rel_width[:w * 2 - 1, :].unsqueeze(0).unsqueeze(0).transpose(2, 3)

        q_with_height = q.permute(2, 0, 1, 3).reshape(c // self.heads, b * self.heads, h * w)
        q_with_width = q.permute(2, 0, 3, 1).reshape(c // self.heads, b * self.heads, h * w)

        height_attn = torch.matmul(rel_height, q_with_height).reshape(-1, h, w * 2 - 1).permute(1, 0, 2)
        width_attn = torch.matmul(rel_width, q_with_width).reshape(-1, w, h * 2 - 1).permute(2, 0, 1)

        attn = content_content_attn + height_attn[:, :, :h] + width_attn[:, :, :w]
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v).reshape(b, self.heads, c // self.heads, h, w).sum(1)
        out = self.attn_out(out)

        return out + x  # Residual connection

# Example usage in a YOLOv8-like model block
class CustomBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, fmap_size, use_attention=False):
        super(CustomBottleneck, self).__init__()
        self.use_attention = use_attention
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if self.use_attention:
            self.attention = BottleneckAttention(out_channels, fmap_size)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.use_attention:
            out = self.attention(out)
        out += identity
        return F.relu(out)

# Example of integrating the attention module into a specific block of YOLOv8
