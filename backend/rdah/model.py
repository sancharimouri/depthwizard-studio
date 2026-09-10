"""RDAH-Net model definition, extracted from the official reference implementation."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvLayer(nn.Module):
    """MobileViT基础卷积块（Conv+BN+激活）"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

def add_colorbar_to_image(image, global_min, global_max, bar_height=20, bar_width=None):
    """
    为图像添加底部色彩条（标注实际高度值）
    Args:
        image: 输入RGB图像（形状 [H, W, 3]）
        global_min: 全局最小高度（实际米数）
        global_max: 全局最大高度（实际米数）
        bar_height: 色彩条高度（像素）
        bar_width: 色彩条宽度（默认与图像宽度一致）
    Returns:
        拼接色彩条后的新图像（形状 [H+bar_height, W, 3]）
    """
    H, W, _ = image.shape
    bar_width = bar_width or W
    # 1. 生成色彩条数据（从 0 到 255 渐变）
    colorbar_data = np.linspace(0, 255, bar_width, dtype=np.uint8)
    colorbar_data = np.tile(colorbar_data, (bar_height, 1))  # 形状：[bar_height, bar_width]
    # 2. 应用与图像相同的色彩映射
    colorbar = cv2.applyColorMap(colorbar_data, cv2.COLORMAP_JET)
    # 3. 拼接图像和色彩条（底部拼接）
    image_with_bar = np.vstack([image, colorbar])
    # 4. 标注实际高度值（最低值在左，最高值在右）
    # 配置标注参数
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_color = (255, 255, 255)  # 白色字体（对比明显）
    thickness = 2
    # 标注最小值（左下角）
    min_text = f"{global_min:.2f} m"  # 保留2位小数，标注单位米
    min_text_size = cv2.getTextSize(min_text, font, font_scale, thickness)[0]
    min_text_x = 10
    min_text_y = H + bar_height - 5  # 色彩条底部上方5像素
    cv2.putText(image_with_bar, min_text, (min_text_x, min_text_y), 
                font, font_scale, font_color, thickness, cv2.LINE_AA)
    # 标注最大值（右下角）
    max_text = f"{global_max:.2f} m"
    max_text_size = cv2.getTextSize(max_text, font, font_scale, thickness)[0]
    max_text_x = W - max_text_size[0] - 10
    max_text_y = H + bar_height - 5
    cv2.putText(image_with_bar, max_text, (max_text_x, min_text_y), 
                font, font_scale, font_color, thickness, cv2.LINE_AA)
    
    return image_with_bar
class BlockAttention(nn.Module):
    """分块注意力（修复维度匹配）"""
    def __init__(self, dim, num_heads=4, block_size=8, mlp_dim=None, dropout=0.):
        super().__init__()
        # 关键：确保num_heads能整除dim，否则报错
        assert dim % num_heads == 0, f"dim={dim} must be an integer multiple of num_heads={num_heads} "
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # 确保整除
        self.block_size = block_size
        self.mlp_dim = mlp_dim or dim * 2
        self.scale = self.head_dim ** -0.5

        self.local_proj = ConvLayer(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.qkv = nn.Conv2d(dim, dim * 3, 1)  # 输出通道=dim×3（q/k/v各占dim）
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, self.mlp_dim, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(self.mlp_dim, dim, 1),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        B, C, H, W = x.shape  # C=dim（确保等于num_heads×head_dim）
        local_feat = self.local_proj(x)

        # 分块确保H/W能被block_size整除
        num_blocks_h = H // self.block_size
        num_blocks_w = W // self.block_size
        num_blocks = num_blocks_h * num_blocks_w

        # 重排为块结构
        x_blocked = x.reshape(
            B, C, num_blocks_h, self.block_size, num_blocks_w, self.block_size
        ).permute(0, 2, 4, 1, 3, 5)  # [B, num_blocks_h, num_blocks_w, C, block_size, block_size]
        x_blocked = x_blocked.reshape(B * num_blocks, C, self.block_size, self.block_size)  # [B×num_blocks, C, 8, 8]

        # 计算qkv
        B_blocked, C_blocked, h, w = x_blocked.shape
        n = h * w  # 8×8=64
        qkv = self.qkv(x_blocked)  # [B_blocked, 3×C, h, w]
        qkv = qkv.reshape(B_blocked, 3, C_blocked, n)  # [B_blocked, 3, C, 64]
        qkv = qkv.permute(1, 0, 2, 3)  # [3, B_blocked, C, 64]
        q, k, v = qkv[0], qkv[1], qkv[2]  # 各为 [B_blocked, C, 64]

        # 拆分多头
        q = q.reshape(B_blocked, self.num_heads, self.head_dim, n)  # [B_blocked, num_heads, head_dim, 64]
        k = k.reshape(B_blocked, self.num_heads, self.head_dim, n)
        v = v.reshape(B_blocked, self.num_heads, self.head_dim, n)

        # 注意力计算
        attn = torch.einsum('bhdn, bhdm -> bhnm', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 注意力输出
        out = torch.einsum('bhnm, bhdm -> bhdn', attn, v)  # [B_blocked, num_heads, head_dim, 64]
        out = out.contiguous().reshape(B_blocked, C_blocked, h, w)  # 还原为[B_blocked, C, 8, 8]
        out = self.proj_drop(self.proj(out))

        # 还原块结构为原始H×W
        out = out.reshape(
            B, num_blocks_h, num_blocks_w, C, self.block_size, self.block_size
        ).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)

        # 融合+MLP
        x = x + local_feat + out
        x = x + self.mlp(x)
        return x

class MobileViTBlock(nn.Module):
    """优化后的MobileViT块"""
    def __init__(self, in_channels, out_channels, stride=1, num_heads=4, block_size=8, dropout=0.):
        super().__init__()
        self.conv1 = ConvLayer(in_channels, out_channels, stride=stride)
        # 确保out_channels能被num_heads整除
        assert out_channels % num_heads == 0, f"out_channels={out_channels} 必须是 num_heads={num_heads} 的整数倍"
        self.attention = BlockAttention(out_channels, num_heads, block_size, dropout=dropout)
        self.conv2 = ConvLayer(out_channels, out_channels, groups=out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.attention(x)
        x = self.conv2(x)
        return x

class MobileViT_S_Light(nn.Module):
    """轻量化MobileViT-S"""
    def __init__(self, in_channels=3):
        super().__init__()
        self.in_channels = in_channels

        # stem层
        self.stem = ConvLayer(in_channels, 32, kernel_size=4, stride=2, padding=1)
        self.stage1 = nn.Sequential(
            # 64通道 → 4头
            MobileViTBlock(32, 64, stride=2, num_heads=4, block_size=8),  # 512→256
            MobileViTBlock(64, 64, stride=1, num_heads=4, block_size=8)
        )
        self.stage2 = nn.Sequential(
            # 128通道 → 8头
            MobileViTBlock(64, 128, stride=2, num_heads=8, block_size=8),  # 256→128
            MobileViTBlock(128, 128, stride=1, num_heads=8, block_size=8)
        )
        self.stage3 = nn.Sequential(
            # 256通道 → 8头
            MobileViTBlock(128, 256, stride=2, num_heads=8, block_size=8),  # 128→64
            MobileViTBlock(256, 256, stride=1, num_heads=8, block_size=8)
        )

        # 投影到d_model=32
        self.proj1 = ConvLayer(64, 32, kernel_size=1, padding=0)
        self.proj2 = ConvLayer(128, 32, kernel_size=1, padding=0)
        self.proj3 = ConvLayer(256, 32, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.stem(x)  # [B,32,512,512]
        feat1 = self.stage1(x)  # [B,64,256,256] → 32通道
        feat2 = self.stage2(feat1)  # [B,128,128,128] →32通道
        feat3 = self.stage3(feat2)  # [B,256,64,64] →32通道

        feat1 = self.proj1(feat1)
        feat2 = self.proj2(feat2)
        feat3 = self.proj3(feat3)

        return [feat1, feat2, feat3]

# -------------------------- 2. 优化跨模态注意力--------------------------
class CBAM(nn.Module):
    """轻量CBAM（保持不变）"""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.spatial = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        x = x * channel_att
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid(self.spatial(torch.cat([avg_out, max_out], dim=1)))
        return x * spatial_att

class PositionalEncoding(nn.Module):
    """位置编码（d_model=32）"""
    def __init__(self, d_model=32, H=64, W=64):
        super().__init__()
        self.d_model = d_model
        pos_x = torch.arange(W, dtype=torch.float32).repeat(H, 1)
        pos_y = torch.arange(H, dtype=torch.float32).repeat(W, 1).t()
        pos = torch.stack([pos_x, pos_y], dim=0)
        pe = torch.zeros(1, d_model, H, W)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[0, ::2, :, :] = torch.sin(pos[0:1, :, :] * div_term[None, :, None, None])
        pe[0, 1::2, :, :] = torch.cos(pos[1:2, :, :] * div_term[None, :, None, None])
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :x.size(2), :x.size(3)]

class LightCrossAttention(nn.Module):
    """轻量化跨模态注意力"""
    def __init__(self, d_model=32, num_heads=4, block_size=8, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model={d_model} must be an integer multiple of num_heads={num_heads}"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.block_size = block_size
        self.scale = self.head_dim ** (-0.5)
        self.dropout = nn.Dropout(dropout)
        self.proj_q = nn.Conv2d(d_model, d_model, 1)
        self.proj_k = nn.Conv2d(d_model, d_model, 1)
        self.proj_v = nn.Conv2d(d_model, d_model, 1)
        self.proj_out = nn.Conv2d(d_model, d_model, 1)
        self.norm = nn.BatchNorm2d(d_model)

    def forward(self, q, k, v):
        B, C, H, W = q.shape  # C=d_model=32
        q_original = q
        num_blocks_h = H // self.block_size
        num_blocks_w = W // self.block_size
        num_blocks = num_blocks_h * num_blocks_w

        # 分块处理
        def blockify(x):
            return x.reshape(
                B, C, num_blocks_h, self.block_size, num_blocks_w, self.block_size
            ).permute(0, 2, 4, 1, 3, 5).reshape(B * num_blocks, C, self.block_size, self.block_size)

        q_blocked = blockify(q)
        k_blocked = blockify(k)
        v_blocked = blockify(v)

        # 计算qkv
        B_blocked, C_blocked, h, w = q_blocked.shape
        n = h * w  # 8×8=64

        q = self.proj_q(q_blocked)  # [B_blocked, 32, 8, 8]
        k = self.proj_k(k_blocked)
        v = self.proj_v(v_blocked)

        # 拆分多头
        q = q.reshape(B_blocked, self.num_heads, self.head_dim, n)  # [B_blocked,4,8,64]
        k = k.reshape(B_blocked, self.num_heads, self.head_dim, n)
        v = v.reshape(B_blocked, self.num_heads, self.head_dim, n)

        # 注意力计算
        attn = torch.einsum('bhdn, bhdm -> bhnm', q, k) * self.scale  # [B_blocked,4,64,64]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum('bhnm, bhdm -> bhdn', attn, v)  # [B_blocked,4,8,64]
        out = out.contiguous().reshape(B_blocked, C_blocked, h, w)  # [B_blocked,32,8,8]
        out = self.proj_out(out)

        # 还原块结构
        out = out.reshape(
            B, num_blocks_h, num_blocks_w, C, self.block_size, self.block_size
        ).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)

        return self.norm(out + q_original)

class LightTransformerBlock(nn.Module):
    """轻量化Transformer"""
    def __init__(self, d_model=32, num_heads=4, hidden_dim=64, block_size=8, dropout=0.1):
        super().__init__()
        self.self_attn = LightCrossAttention(d_model, num_heads, block_size, dropout)
        self.ffn = nn.Sequential(
            nn.Conv2d(d_model, hidden_dim, 1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, d_model, 1)
        )
        self.norm = nn.BatchNorm2d(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.self_attn(x, x, x)
        x = x + self.dropout(self.ffn(self.norm(x)))
        return x

    
class HeightPredTransformer(nn.Module):
    def __init__(self, d_model=32, num_heads=4):  # d_model从64→32，num_heads从8→4
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        # self.H, self.W = 1024, 1024

        # 1. 轻量化MobileViT-S双分支编码器
        self.depth_encoder = MobileViT_S_Light(in_channels=1)
        self.img_encoder = MobileViT_S_Light(in_channels=3)

        # 2. CBAM注意力
        self.cbam_blocks = nn.ModuleList([
            CBAM(d_model),
            CBAM(d_model),
            CBAM(d_model)
        ])

        # 3. 轻量化跨模态融合
        self.cross_attn_blocks = nn.ModuleList([
            LightCrossAttention(d_model, num_heads, block_size=8),
            LightCrossAttention(d_model, num_heads, block_size=8),
            LightCrossAttention(d_model, num_heads, block_size=8)
        ])
        self.rev_cross_attn_blocks = nn.ModuleList([
            LightCrossAttention(d_model, num_heads, block_size=8),
            LightCrossAttention(d_model, num_heads, block_size=8),
            LightCrossAttention(d_model, num_heads, block_size=8)
        ])

        # 4. 轻量化全局Transformer
        self.pos_encoding = PositionalEncoding(d_model, H=64, W=64)
        self.global_transformer = LightTransformerBlock(d_model, num_heads, hidden_dim=64)

        # 5. 轻量化解码器（降通道）
        self.skip_projs = nn.ModuleList([
            nn.Conv2d(d_model, 16, 1),  # 32→16
            nn.Conv2d(d_model, 32, 1),  # 32→32
            nn.Conv2d(d_model, 64, 1)   # 32→64
        ])

        self.decoder = nn.Sequential(
            # 64×64 → 128×128（降通道：128×4→64×4）
            nn.Conv2d(d_model, 64 * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 128×128 → 256×256（64×4→32×4）
            nn.Conv2d(64, 32 * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 256×256 → 512×512（32×4→16×4）
            nn.Conv2d(32, 16 * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # 512×512 → 1024×1024（简化ESPCN）
            nn.Conv2d(16, 8 * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 1, 3, padding=1)  # 去掉多余的ESPCN层，简化解码
        )

    def forward(self, depth, img):
        B = depth.shape[0]
        # assert depth.shape == (B, 1, self.H, self.W), f"深度图形状错误！需 [B,1,1024,1024]，当前 {depth.shape}"
        # assert img.shape == (B, 3, self.H, self.W), f"影像形状错误！需 [B,3,1024,1024]，当前 {img.shape}"

        # 步骤1：提取多尺度特征
        depth_feats = self.depth_encoder(depth)
        img_feats = self.img_encoder(img)

        # CBAM增强
        depth_feats = [self.cbam_blocks[i](f) for i, f in enumerate(depth_feats)]
        img_feats = [self.cbam_blocks[i](f) for i, f in enumerate(img_feats)]

        # 步骤2：层级化双向融合
        fused_feats = []
        for i in range(3):
            feat1 = self.cross_attn_blocks[i](q=depth_feats[i], k=img_feats[i], v=img_feats[i])
            feat2 = self.rev_cross_attn_blocks[i](q=img_feats[i], k=depth_feats[i], v=depth_feats[i])
            fused_feat = (feat1 + feat2) / 2
            fused_feats.append(fused_feat)

        # 全局Transformer
        global_fused_feat = self.pos_encoding(fused_feats[2])
        global_fused_feat = self.global_transformer(global_fused_feat)

        # 步骤3：解码
        x = global_fused_feat
        # 阶段1：下采样8倍 → 下采样4倍（动态获取当前输出尺寸）
        x = self.decoder[0:4](x)
        target_size1 = x.shape[2:]  # 替代硬编码的(128,128)
        skip_feat = self.skip_projs[2](fused_feats[2])
        skip_feat = F.interpolate(skip_feat, size=target_size1, mode='bilinear', align_corners=False)
        x = x + skip_feat

        # 阶段2：下采样4倍 → 下采样2倍
        x = self.decoder[4:8](x)
        target_size2 = x.shape[2:]  # 替代硬编码的(256,256)
        skip_feat = self.skip_projs[1](fused_feats[1])
        skip_feat = F.interpolate(skip_feat, size=target_size2, mode='bilinear', align_corners=False)
        x = x + skip_feat

        # 阶段3：下采样2倍 → 输入尺寸
        x = self.decoder[8:12](x)
        target_size3 = x.shape[2:]  # 替代硬编码的(512,512)
        skip_feat = self.skip_projs[0](fused_feats[0])
        skip_feat = F.interpolate(skip_feat, size=target_size3, mode='bilinear', align_corners=False)
        x = x + skip_feat

        # 最终解码：输入尺寸 → 2×输入尺寸（保留原有逻辑）
        height_pred = self.decoder[12:](x)

        return height_pred

    
# 定义带掩码的损失函数

