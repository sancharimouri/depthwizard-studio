import warnings
warnings.filterwarnings('ignore')  # 忽略所有警告
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import argparse 
import re
import math
import glob
import os
import rasterio
from datetime import datetime
import tifffile
import torch.nn.init as init
import torch.nn.functional as F
from rasterio.errors import RasterioIOError
import loaddata
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class UpSampleConv(nn.Module):
    def __init__(self, in_c, out_c, scale_factor=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(self.upsample(x))

class ConvLayer(nn.Module):
    """MobileViT基础卷积块（Conv+BN+激活）"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class BlockAttention(nn.Module):
    """分块注意力（修复维度匹配）"""
    def __init__(self, dim, num_heads=4, block_size=8, mlp_dim=None, dropout=0.):
        super().__init__()
        # 关键：确保num_heads能整除dim，否则报错
        assert dim % num_heads == 0, f"dim={dim} must be an integer multiple of num_heads={num_heads}"
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
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B_blocked, C, 64]
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
    """优化后的MobileViT块（确保维度匹配）"""
    def __init__(self, in_channels, out_channels, stride=1, num_heads=4, block_size=8, dropout=0.):
        super().__init__()
        self.conv1 = ConvLayer(in_channels, out_channels, stride=stride)
        # 确保out_channels能被num_heads整除
        assert out_channels % num_heads == 0, f"out_channels={out_channels} must be an integer multiple of num_heads={num_heads}"
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

        # stem层32通道
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
    """轻量CBAM"""
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
    # 位置编码
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
    """轻量化跨模态注意力（修复维度匹配）"""
    def __init__(self, d_model=32, num_heads=4, block_size=8, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model={d_model} Must be an integer multiple of num_heads={num_heads}"
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
        n = h * w  
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
    # """轻量化Transformer"""
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
    def __init__(self, d_model=32, num_heads=4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        # self.H, self.W = 512, 512  # 注释掉硬编码尺寸

        # 1. 轻量化MobileViT-S双分支编码器
        self.depth_encoder = MobileViT_S_Light(in_channels=1)
        self.img_encoder = MobileViT_S_Light(in_channels=3)

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
        self.pos_encoding = PositionalEncoding(d_model)  
        self.global_transformer = LightTransformerBlock(d_model, num_heads, hidden_dim=64)

        # 5. 轻量化解码器
        self.skip_projs = nn.ModuleList([
            nn.Conv2d(d_model, 16, 1),
            nn.Conv2d(d_model, 32, 1),
            nn.Conv2d(d_model, 64, 1)
        ])

        # 解码器结构
        self.decoder = nn.Sequential(
            nn.Conv2d(d_model, 64 * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32 * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16 * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8 * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 1, 3, padding=1)
        )

    def forward(self, depth, img):
        B = depth.shape[0]
        # 1. 动态获取输入尺寸
        _, _, H_in, W_in = depth.shape
        # 校验通道数
        assert depth.shape[1] == 1, f"Depth Channel invalid! Only 1 channel needed, currently {depth.shape[1]}"
        assert img.shape == (B, 3, H_in, W_in), f"Image Shape invalid![B,3,{H_in},{W_in}]needed, currently{img.shape}"
        # 2. 提取多尺度特征
        depth_feats = self.depth_encoder(depth)
        img_feats = self.img_encoder(img)
        depth_feats = [self.cbam_blocks[i](f) for i, f in enumerate(depth_feats)]
        img_feats = [self.cbam_blocks[i](f) for i, f in enumerate(img_feats)]
        # 3. 层级化双向融合
        fused_feats = []
        for i in range(3):
            feat1 = self.cross_attn_blocks[i](q=depth_feats[i], k=img_feats[i], v=img_feats[i])
            feat2 = self.rev_cross_attn_blocks[i](q=img_feats[i], k=depth_feats[i], v=depth_feats[i])
            fused_feat = (feat1 +feat2)/2
            fused_feats.append(fused_feat)

        # 4. 全局Transformer（动态位置编码）
        global_fused_feat = self.pos_encoding(fused_feats[2])
        global_fused_feat = self.global_transformer(global_fused_feat)
        # 5. 解码
        x = global_fused_feat
        x = self.decoder[0:4](x)
        target_size1 = x.shape[2:]  
        skip_feat = self.skip_projs[2](fused_feats[2])
        skip_feat = F.interpolate(skip_feat, size=target_size1, mode='bilinear', align_corners=False)
        x = x + skip_feat

        x = self.decoder[4:8](x)
        target_size2 = x.shape[2:]  
        skip_feat = self.skip_projs[1](fused_feats[1])
        skip_feat = F.interpolate(skip_feat, size=target_size2, mode='bilinear', align_corners=False)
        x = x + skip_feat

        x = self.decoder[8:12](x)
        target_size3 = x.shape[2:]  
        skip_feat = self.skip_projs[0](fused_feats[0])
        skip_feat = F.interpolate(skip_feat, size=target_size3, mode='bilinear', align_corners=False)
        x = x + skip_feat
        height_pred = self.decoder[12:](x)

        # 若希望输出尺寸与输入一致，可添加这行
        # height_pred = F.interpolate(height_pred, size=(H_in, W_in), mode='bilinear', align_corners=False)

        return height_pred
    
class MaskedLoss(nn.Module):
    def __init__(self, base_loss=torch.nn.SmoothL1Loss(reduction='none')):
        super(MaskedLoss, self).__init__()
        self.base_loss = base_loss
        
    def forward(self, pred, target, mask=None):
        if mask is None:
            mask = torch.ones_like(target, dtype=torch.float32)
        else:
            mask = mask.float()
        # 有效区域掩码
        target = torch.where(torch.isnan(target), pred, target)
        valid = (~torch.isnan(target)) & (mask > 0)
      
        nan_penalty = torch.tensor(1e4, device=pred.device) * torch.isnan(pred).float().sum()
    
        # 计算基础损失
        loss = self.base_loss(pred, target)
        masked_loss = loss.where(valid.bool(), torch.tensor(0.0, device=loss.device))

        # 计算平均损失
        num_valid = valid.sum()
        if num_valid == 0:
            return torch.tensor(0.0, device=loss.device)

        loss_final = masked_loss.sum() / num_valid
        
        loss_final = loss_final + nan_penalty

        return loss_final
def broad(tensor):
    s = tensor.shape[0]
    # print(s)
    return tensor.view(s, 1, 1, 1)


def test_model(dataloader_val, model, savepic= False, device="cuda", save_root='./checkpoints-v2-Swiss'):
    # 测试模型
    model.eval()
    model = model.to(device)  
    total_loss = 0.0
    num_samples = len(dataloader_val)
    criterion = MaskedLoss().to(device) 
    with torch.no_grad():
        for i, sample in enumerate(dataloader_val):
            depth_input = sample['rel_depth'].to(device)  
            image_input = sample['image'].to(device)
            
            prediction = model(depth_input, image_input)
            # sample['depth'] = prediction
            
            # 计算 loss
            loss = criterion(prediction, sample['depth'].to(device))
            total_loss += loss.item()
            
    # 计算平均 loss
    avg_loss = total_loss / num_samples
    print(f"Test set Average Loss: {avg_loss:.4f}")    
    # 可选：保存平均 loss 到文件
    os.makedirs(save_root, exist_ok=True)
    with open(os.path.join(save_root, 'test_avg_loss.txt'), 'w') as f:
        f.write(f"Test set Average Loss: {avg_loss:.4f}\n")
     # 检查参数是否有 NaN
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"NaN detected in parameter {name} at batch {i}")
            exit()
    return avg_loss
# 训练函数
def train_model(pretrained, timestamp, train_pred='', train_depth='', train_gt='', val_pred='', val_depth='', val_gt='', num_epochs=1, batch_size=2, learning_rate=1e-5, save_root='./checkpoints-v2-Swiss', save_interval=5):
    
    save_path = os.path.join(save_root, timestamp)
    os.makedirs(save_path, exist_ok=True)
    # 创建数据加载器
    dataloader_train = loaddata.getTrainingData(batch_size, train_pred, train_depth, train_gt)
    dataloader_val = loaddata.getTestingData(batch_size, val_pred, val_depth, val_gt)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}") 
    
    model = HeightPredTransformer().to(device) 
    criterion = MaskedLoss().to(device)     
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # load pretrained 
    if pretrained is not None:
        checkpoint = torch.load(pretrained, map_location='cuda' if torch.cuda.is_available() else 'cpu')

        # 还原模型权重
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Successfully loaded model weights from checkpoint")
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # 还原epoch和loss
        start_epoch = checkpoint['epoch']  
        best_loss = checkpoint['loss']

        print(f"Continue training: From epoch {start_epoch}，Pre-trained last loss={best_loss:.4f}")
    model.train()
    # 统计总参数量（包括不可训练参数，如 BatchNorm 的 running_mean/running_var）
    total_params = sum(p.numel() for p in model.parameters())
# 统计可训练参数量（仅需要更新的参数）
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("total params")
    print(total_params)
    print("trainable params")
    print(trainable_params)
    losses = []
    best_loss = float('inf')  
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for batch in dataloader_train:
            # 获取输入
            depths = batch['rel_depth']
            heights = batch['depth']
            images = batch['image']
            masks = batch['mask']
            depths = depths.to(device)    
            heights = heights.to(device)  
            images = images.to(device)
            masks = masks.to(device)     
            if torch.isnan(depths).any():
                print("input has NaN!")
            # 清零梯度
            optimizer.zero_grad()
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad.zero_()
            # 前向传播
            outputs = model(depths, images)
            loss = criterion(outputs, heights, masks)+ 1e-4 * torch.norm(outputs, p=2)
            loss.backward()
            
            for name, param in model.named_parameters():
                if param.grad is not None :
                    # print(param.grad)
                    if (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                        print(f"Gradient explosion in {name}")
                        exit()
            optimizer.step()
            
            # print(loss.item())
            running_loss += loss.item() * depths.size(0)
        
        # 计算 epoch 平均损失
        epoch_loss = running_loss / len(train_pred)
        losses.append(epoch_loss)
        
        # 保存最佳模型
        savepic = True if epoch == num_epochs else False
        epoch_test_loss = test_model(dataloader_val, model, savepic, device, save_root=save_root)
        if epoch_test_loss < best_loss:
            best_loss = epoch_test_loss
            best_checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss
            }
            torch.save(best_checkpoint, f'{save_path}/best_model.pth')
            print(f'Best model saved (Loss: {best_loss:.4f}) at epoch {epoch+1}')
            
    # 保存最终模型
    torch.save(model.state_dict(), f'{save_path}/final_model.pth')
    print(f'Final model saved: {save_path}/final_model.pth')
    
    # 绘制损失曲线
    # plt.plot(losses)
    plt.plot(range(1, len(losses)+1), losses)  # x轴：1,2,...,len(losses)；y轴：losses值
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xlim(1, len(losses))
    # 保存为 TIF 格式
    plt.savefig('training_loss'+str(epoch)+'.png', format='png', dpi=300)
    plt.close() 
    # plt.show()
    return model
def test_all(args_height_all, timestamp, dataset, model, criterion, device, save_dir='./test_results'):
    """
    测试整个数据集并计算平均 loss，可选保存每张结果图
    """
    model.eval()
    model = model.to(device)  # 确保模型在GPU上
    total_loss = 0.0
    num_samples = len(dataset)
    save_dir = os.path.join(save_dir, timestamp)
    mean, std = args_height_all
    os.makedirs(save_dir, exist_ok=True)  # 创建保存结果的文件夹
    
    with torch.no_grad():  # 关闭梯度计算
        for idx in range(num_samples):
            sample = dataset[idx]
            # 准备输入
            depth_input = sample['depth'].unsqueeze(0).to(device)   
            target_height = sample['height'].unsqueeze(0).to(device) 
            mask = sample['mask'].unsqueeze(0).bool().to(device) 
            height_mean, height_std = sample['args_height']
            # 模型预测
            prediction = model(depth_input)
            prediction = prediction* std + mean
            target_height = target_height * height_std + height_mean
            # 计算 loss
            mae = torch.mean(torch.abs(target_height[mask] - prediction[mask]))

            total_loss += mae.item()
            print(f"Sample {idx+1}/{num_samples}, Loss: {mae.item():.4f}")
            # 可视化
            plt.figure(figsize=(15, 5))
            
            plt.subplot(131)
            plt.imshow(sample['depth'].cpu().numpy()[0], cmap='viridis')
            plt.title('Relative Depth')
            
            plt.subplot(132)
            plt.imshow(sample['height'].cpu().numpy()[0], cmap='viridis')
            plt.title('True Absolute Height')
            
            plt.subplot(133)
            plt.imshow(prediction.cpu().numpy()[0, 0], cmap='viridis')
            plt.title('Predicted Absolute Height')
            
            plt.tight_layout()
            # 保存图片
            save_path = os.path.join(save_dir, f'test_result_{idx}.png')
            plt.savefig(save_path, format='png', dpi=300)
            plt.close()  
            true_height_np = sample['height'].squeeze().cpu().numpy()  # shape: (H, W)
            pred_height_np = prediction.squeeze().cpu().numpy()     # shape: (H, W)
            # 保存路径
            true_tif_path = os.path.join(save_dir, f'true_height_{idx}.tif')
            pred_tif_path = os.path.join(save_dir, f'pred_height_{idx}.tif')
            # 保存为TIFF
            tifffile.imwrite(
                true_tif_path,
                true_height_np,
                dtype=true_height_np.dtype,  
                compression=None  
            )
            tifffile.imwrite(
                pred_tif_path,
                pred_height_np,
                dtype=pred_height_np.dtype,
                compression=None
            )
    # 计算平均 loss
    avg_loss = total_loss / num_samples
    print(f"\nTest set Average Loss: {avg_loss:.4f}")
    
    # 保存平均 loss 到文件
    with open(os.path.join(save_dir, 'test_avg_loss.txt'), 'w') as f:
        f.write(f"Test set Average Loss: {avg_loss:.4f}\n")
    
    return avg_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Folder Path, checkpoint path and training parameters")
    parser.add_argument('--input_folder', type=str, required=True, help='Input training set Folder Path or txt file path')
    parser.add_argument('--save_root', type=str, required=True, help='Root folder to save checkpoints')
    parser.add_argument('--prefix_gt', type=str, required=True, help='Optional prefix for ground truth files')
    parser.add_argument('--prefix_depth', type=str, required=True, help='Optional prefix for Depth files')
    parser.add_argument('--prefix_image', type=str, required=True, help='Optional prefix for Image files')

    parser.add_argument('--channel', type=int, required=True, help='channel')
    parser.add_argument('--epoch', type=int, required=True, help='training epochs')
    parser.add_argument('--H', type=int, required=True, help='H')
    parser.add_argument('--W', type=int, required=True, help='W')
    parser.add_argument('--load_check', type=str, default=None, help='load pretrained checkpoint and optimizer, continue train')

    args = parser.parse_args()

    if args.input_folder.endswith('.txt'):
    # 如果输入是txt文件，读取每行的tif路径
        head_path = os.path.dirname(args.input_folder)
      
        gt_list = []
        pred_list = []
        with open(args.input_folder, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()           
                if not line:
                    continue
                parts = line.split(' ', 1)
                # 确保每行确实有两个元素
                if len(parts) == 2:
                    pred, gt = parts
                    pred_list.append(pred)
                    gt_list.append(gt)
                elif len(parts) == 1:
                    pred = parts[0]
                    pred_list.append(pred)
                else:
                    print(f"Warning: Line '{line}' Format Error, skip")
        
        pred_list_ori = pred_list
        pred_list = [os.path.join(args.prefix_depth,f) for f in pred_list]
        gt_list = [os.path.join(args.prefix_gt,f.replace('Ortho','nDSM')) for f in pred_list_ori]
        img_list = [os.path.join(args.prefix_image,f) for f in pred_list_ori]
        for p, d, g in zip(pred_list, gt_list, img_list):
            # 检查三个文件是否都存在
            if os.path.exists(p)==False or os.path.exists(d)==False or os.path.exists(g)==False:
                print(f"⚠️  Invalid sample, skip：p={p}, d={d}, g={g}(file not exsits))")        
    else:
        # 否则从文件夹中查找tif文件
        tif_files = sorted(glob.glob(os.path.join(args.input_folder, '*.tif')))
        gt_list = []
        pred_list = []
        for f in tif_files:
            if 'gt' in os.path.basename(f) or 'groundtruth' in os.path.basename(f).lower():
                gt_list.append(f)
            else:
                pred_list.append(f)
    N = len(gt_list)
    C = args.channel
    H = args.H
    W = args.W
    gt_batch = np.empty((N, C, H, W), dtype=float)
    pred_batch = np.empty((N, C,H, W), dtype=float)
    gt_list.sort()
    pred_list.sort()
    assert len(gt_list)==len(pred_list)

    train_ratio = 0.8
    N = gt_batch.shape[0]  # 总样本数
    all_indices = np.arange(N)  # [0, 1, 2, ..., N-1]
    np.random.shuffle(all_indices)

    # 计算划分边界（train取前80%，val取后20%）
    train_size = int(N * train_ratio)
    train_indices = all_indices[:train_size]  
    val_indices = all_indices[train_size:]    
    gt_list = np.array(gt_list)
    pred_list = np.array(pred_list)
    img_list = np.array(img_list)
    train_gt = gt_list[np.array(train_indices)]   
    train_depth = pred_list[np.array(train_indices)]
    train_pred = img_list[np.array(train_indices)]
    val_gt = gt_list[val_indices]      
    val_depth = pred_list[val_indices]
    val_pred = img_list[val_indices]

    # 输出划分结果（验证是否正确）
    print(f"Total Sample Number: {N}")
    print(f"Number of training samples: {len(train_indices)}, Shape: {train_gt.shape}")
    print(f"Number of validation samples: {len(val_indices)}, Shape: {val_gt.shape}")
    

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # 训练模型
    print("model going to start")
    model, args_height = train_model(args.load_check, timestamp, train_pred, train_depth, train_gt, val_pred, val_depth, val_gt, num_epochs=args.epoch, batch_size=2, save_root=args.save_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 测试模型
    # test_all(args_height,timestamp, valset, model, MaskedLoss(), device, save_dir='./test_results')

