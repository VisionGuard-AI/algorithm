import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_

# Patch Embedding
class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding, from timm

    图像到 Patch 嵌入模块（Patch Embedding）：
    将输入图像划分为固定大小的 patch，并将每个 patch 映射到高维嵌入空间。

    structure:
        proj(Conv2d) - flatten
    """
    def __init__(self, img_size=64, patch_size=16, in_chans=3, embed_dim=768):
        """
        初始化 PatchEmbed 模块。

        Args：
            :param img_size : 输入图像的尺寸，默认值为 64
            :param patch_size : 每个补丁的大小，默认值为 16
            :param in_chans : 输入图像的通道数，默认值为 3
            :param embed_dim : 每个补丁的嵌入维度，默认值为 768
        """
        super().__init__()

        # 计算图像被划分的补丁总数
        num_patches = (img_size // patch_size) * (img_size // patch_size)

        # 保存图像尺寸、补丁大小和补丁数量
        self.img_size = img_size    # 图像尺寸
        self.patch_size = patch_size    # 补丁大小
        self.num_patches = num_patches      # 补丁数量

        # 定义 2D 卷积层，用于将图像划分为补丁并映射到嵌入空间
        # kernel_size 和 stride 都设置为 patch_size，以确保每个补丁不重叠
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        # # 初始化权重
        # self.apply(self._init_weights)

    def forward(self, x):
        """
        前向传播逻辑：
        将输入图像划分为 patch，并将每个 patch 嵌入高维空间。

        Args：
            :param x (torch.Tensor): 输入张量，形状为 [batch_size, in_chans, img_height, img_width]。
            :param batch_size: 批量大小。
            :param in_chans: 输入图像的通道数。
            :param img_height: 输入图像的高度。
            :param img_width: 输入图像的宽度。

        :return
            torch.Tensor: 输出张量，形状为 [batch_size, num_patches, embed_dim]。
        """
        # 获取输入张量的形状
        B, C, H, W = x.shape        # B: 批量大小, C: 通道数, H: 图像高度, W: 图像宽度
        # 验证输入图像的尺寸是否与模型预期的一致
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})."

        # 通过卷积层将图像划分为补丁并映射到嵌入空间
        # 输出形状为 [batch_size, embed_dim, num_patches_height, num_patches_width]
        # 展平补丁维度（保留 batch 和嵌入维度）
        # flatten(2): 将 [embed_dim, num_patches_height, num_patches_width] 展平为 [embed_dim, num_patches]
        # transpose(1, 2): 调整形状为 [batch_size, num_patches, embed_dim]
        x = self.proj(x).flatten(2).transpose(1, 2)

        return x

# transformer block
class Block(nn.Module):
    """
    Transformer block：
        包含多头注意力机制（MHSA 或 GPSA）、多层感知机（MLP）、归一化层和残差连接。

    structure:
        norm1(LayerNorm) - attn(GPSA/MHSA) - drop_path(DropPath) - norm2(LayerNorm) - mlp(MLP) - drop_path(DropPath)
    """

    def __init__(self, dim, num_heads,  mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_gpsa=True, **kwargs):
        """
        初始化

        Args：
            :param dim (int): 输入特征的维度。
            :param num_heads (int): 注意力头的数量。
            :param mlp_ratio (float): MLP 中隐藏层维度与输入特征维度的比例，默认为 4。
            :param qkv_bias (bool): 是否为 Q、K、V 的线性层添加偏置项，默认值为 False。
            :param qk_scale (float, 可选): Q 和 K 的缩放因子。如果未指定，默认使用 `head_dim ** -0.5`。
            :param drop (float): MLP 输出和注意力权重的 Dropout 概率，默认值为 0。
            :param attn_drop (float): 注意力权重的 Dropout 概率，默认值为 0。
            :param drop_path (float): 随机深度（Stochastic Depth）的丢弃概率，默认值为 0。
            :param act_layer (nn.Module): 激活函数类，默认使用 GELU。
            :param norm_layer (nn.Module): 归一化层类，默认使用 LayerNorm。
            :param use_gpsa (bool): 是否使用 GPSA（Gated Positional Self-Attention），默认值为 True。
            :param **kwargs: 传递给注意力模块（GPSA 或 MHSA）的额外参数。
        """
        super().__init__()

        # 初始化 norm
        self.norm1 = norm_layer(dim)

        # 初始化 注意力机制（GPSA / MHSA）

        # 选择注意力机制
        self.use_gpsa = use_gpsa
        if self.use_gpsa:
            self.attn = GPSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, **kwargs)
        else:
            self.attn = MHSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, **kwargs)

        # 初始化 DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # 初始化 norm
        self.norm2 = norm_layer(dim)

        # 初始化 MLP
        mlp_hidden_dim = int(dim * mlp_ratio)       # hidden dimension
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        """
        forward：
            输入经过归一化、注意力机制、残差连接、MLP 和 DropPath，输出最终的特征。

        Args：
            :param x (torch.Tensor): 输入张量，形状为 [batch_size, num_patches, dim]。

        return
            x (torch.Tensor): 输出张量，形状为 [batch_size, num_patches, dim]。
        """
        # attention branch
        # 输入先经过归一化，再传入注意力机制；然后通过残差连接与原始输入相加。
        x = x + self.drop_path(self.attn(self.norm1(x)))

        # MLP branch
        # 输入经过第二次归一化，再传入 MLP；然后通过残差连接与上一步的输出相加。
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

# MLP
class Mlp(nn.Module):
    """
    MLP block，处理特征变换和非线性激活。

    structure:
        fc1(Linear) - act(GELU) - drop(Dropout) - fc2(Linear) - drop(Dropout)
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        """
        初始化

        Args：
            :param in_features (int): 输入特征的维度。
            :param hidden_features (int, 可选): 隐藏层的特征维度。如果未指定，默认与输入特征维度一致。
            :param out_features (int, 可选): 输出特征的维度。如果未指定，默认与输入特征维度一致。
            :param act_layer (nn.Module, 可选): 激活函数类，默认使用 GELU（高斯误差线性单元）。
            :param drop (float, 可选): Dropout 概率，用于防止过拟合，默认值为 0（不使用 Dropout）。
        """
        super().__init__()

        # 如果未指定输出或隐藏层维度，则默认与输入维度一致。
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # 定义第一个线性层，从输入特征维度映射到隐藏层维度。
        self.fc1 = nn.Linear(in_features, hidden_features)
        # 激活函数层，默认是 GELU，用于引入非线性。
        self.act = act_layer()
        # 定义第二个线性层，从隐藏层维度映射到输出特征维度。
        self.fc2 = nn.Linear(hidden_features, out_features)
        # Dropout 层，用于在训练过程中随机丢弃部分神经元，减少过拟合。
        self.drop = nn.Dropout(drop)

        # 初始化模型权重。
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
                初始化权重的方法，根据不同的层类型进行权重初始化。

                参数：
                - m (nn.Module): 神经网络模块，用于检查其类型并进行对应的初始化。
        """
        if isinstance(m, nn.Linear):
            # 对线性层权重使用截断正态分布初始化，标准差为 0.02。
            trunc_normal_(m.weight, std=.02)
            # 如果线性层有偏置项，将其初始化为 0。
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            # 对 LayerNorm 的偏置项初始化为 0，权重初始化为 1。
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
                定义前向传播逻辑。

                参数：
                - x (torch.Tensor): 输入张量，形状为 [batch_size, seq_len, in_features]。

                返回：
                - torch.Tensor: 输出张量，形状为 [batch_size, seq_len, out_features]。
        """
        # 应用第一个线性变换，将输入特征映射到隐藏层特征维度。
        x = self.fc1(x)
        # 应用激活函数，增加非线性建模能力。
        x = self.act(x)
        # 应用 Dropout，用于随机丢弃部分特征。
        x = self.drop(x)
        # 应用第二个线性变换，将隐藏层特征映射到输出特征维度。
        x = self.fc2(x)
        # 再次应用 Dropout，进一步增加正则化效果。
        x = self.drop(x)
        return x

# GPSA
class GPSA(nn.Module):
    """
        GPSA (Gated Positional Self-Attention) 模块：
        一种增强局部性的自注意力机制，结合了标准的多头自注意力和基于位置编码的局部增强特性。

        structure:
        qk(Linear) - Softmax - Sigmoid - attn_drop(Dropout) - v(Linear) - proj(Linear) - proj_drop(Dropout)

    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 locality_strength=1., use_local_init=True):
        """
                初始化 GPSA 模块。

                参数：
                - dim (int): 输入特征的维度。
                - num_heads (int): 注意力头的数量，默认值为 8。
                - qkv_bias (bool): 是否为 Q、K 和 V 的线性层添加偏置，默认值为 False。
                - qk_scale (float, 可选): Q 和 K 的缩放因子。如果未指定，使用 `head_dim**-0.5`。
                - attn_drop (float): 注意力分数的 Dropout 概率。
                - proj_drop (float): 输出投影的 Dropout 概率。
                - locality_strength (float): 增强局部性的强度。
                - use_local_init (bool): 是否使用基于局部性的初始化，默认值为 True。
        """
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads  # 每个头的维度
        self.scale = qk_scale or head_dim ** -0.5  # Q 和 K 的缩放因子

        # 定义线性层，用于生成 Q 和 K（qk）以及 V。
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)  # Q 和 K 的线性变换
        self.v = nn.Linear(dim, dim, bias=qkv_bias)  # V 的线性变换

        # 注意力和投影的 Dropout 层
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)  # 输出投影层
        self.pos_proj = nn.Linear(3, num_heads)  # 基于相对位置编码的投影
        self.proj_drop = nn.Dropout(proj_drop)

        # 局部性参数和控制门
        self.locality_strength = locality_strength
        self.gating_param = nn.Parameter(torch.ones(self.num_heads))  # 每个头的门控参数

        # 初始化权重
        self.apply(self._init_weights)
        if use_local_init:
            self.local_init(locality_strength=locality_strength)

    def _init_weights(self, m):
        """
                初始化模型权重。

                参数：
                - m (nn.Module): 神经网络模块。
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)  # 截断正态分布初始化
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)  # 偏置初始化为 0
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
                前向传播逻辑。

                参数：
                - x (torch.Tensor): 输入张量，形状为 [batch_size, num_patches, dim]。

                返回：
                - torch.Tensor: 输出张量，形状为 [batch_size, num_patches, dim]。
        """
        B, N, C = x.shape  # B: 批量大小, N: 补丁数量, C: 特征维度
        # 如果相对位置索引未初始化或形状与当前补丁数量不匹配，则重新计算。
        if not hasattr(self, 'rel_indices') or self.rel_indices.size(1) != N:
            self.get_rel_indices(N)

        # 计算注意力分数
        attn = self.get_attention(x)

        # 将输入 x 通过 V 的线性变换后 reshape，变为 [B, num_heads, N, head_dim]
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # 通过注意力分数加权后还原形状
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # 应用输出投影和 Dropout
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def get_attention(self, x):
        """
                计算注意力分数。

                参数：
                - x (torch.Tensor): 输入张量。形状为 [batch_size, num_patches, dim]。

                返回：
                - torch.Tensor: 注意力分数，形状为 [batch_size, num_heads, num_patches, num_patches]。
        """
        B, N, C = x.shape  # B: 批量大小, N: 补丁数量, C: 特征维度

        # Step 1: 通过线性层生成 Q 和 K
        # `qk` 的形状为 [batch_size, num_patches, 2, num_heads, head_dim]
        qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]  # 将 Q 和 K 分离
        # q 和 k 的形状为 [batch_size, num_heads, num_patches, head_dim]

        # Step 2: 计算相对位置分数
        # `self.rel_indices` 存储了补丁之间的相对位置索引，形状为 [1, num_patches, num_patches, 3]
        # 通过 `expand` 扩展到 [batch_size, num_patches, num_patches, 3]
        pos_score = self.rel_indices.expand(B, -1, -1, -1)
        # 通过线性层 `self.pos_proj` 将相对位置索引映射到每个注意力头，结果形状为 [batch_size, num_heads, num_patches, num_patches]
        pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)

        # Step 3: 计算全局内容注意力分数
        # 通过矩阵乘法计算 Q 和 K 的点积，得到形状为 [batch_size, num_heads, num_patches, num_patches]
        patch_score = (q @ k.transpose(-2, -1)) * self.scale
        # 使用 Softmax 对最后一个维度（补丁间）进行归一化
        patch_score = patch_score.softmax(dim=-1)

        # Step 4: 对位置分数进行 Softmax 归一化
        pos_score = pos_score.softmax(dim=-1)

        # Step 5: 合并全局内容注意力和局部位置注意力
        # 使用门控机制控制两种注意力的融合比例
        # `self.gating_param` 是一个形状为 [num_heads] 的可学习参数
        gating = self.gating_param.view(1, -1, 1, 1)  # 调整为 [1, num_heads, 1, 1] 便于广播
        # 通过 Sigmoid 激活函数将门控参数限制在 [0, 1] 范围内
        attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score

        # Step 6: 再次归一化注意力分数，确保每行的和为 1
        attn /= attn.sum(dim=-1).unsqueeze(-1)

        # Step 7: 对注意力分数应用 Dropout，增强正则化效果
        attn = self.attn_drop(attn)

        return attn

    def get_attention_map(self, x, return_map=False):
        """
            计算注意力分布图和每个注意力头的平均距离。

            参数：
            - x (torch.Tensor): 输入张量，形状为 [batch_size, num_patches, dim]。
              - batch_size: 批量大小。
              - num_patches: 图像被划分的补丁数量。
              - dim: 输入特征的维度。
            - return_map (bool): 是否返回完整的注意力分布图（可视化用），默认为 False。

            返回：
            - dist (torch.Tensor): 每个注意力头的平均距离，形状为 [num_heads]。
              - 表示每个头的注意力中心与补丁位置的平均距离。
            - attn_map (torch.Tensor, 可选): 注意力分布图，形状为 [num_heads, num_patches, num_patches]。
              - 仅当 `return_map=True` 时返回。
        """

        # Step 1: 获取注意力分布图
        # 调用 `get_attention` 方法，计算注意力分数，形状为 [batch_size, num_heads, num_patches, num_patches]。
        attn_map = self.get_attention(x).mean(
            0)  # average over batch(对 batch 维度取平均，形状变为 [num_heads, num_patches, num_patches]。)

        # Step 2: 获取补丁之间的相对距离
        # `self.rel_indices` 是形状为 [1, num_patches, num_patches, 3] 的相对位置索引。
        # 其中最后一维的第三个通道（索引为 -1）表示补丁间的平方距离。
        distances = self.rel_indices.squeeze()[:, :, -1] ** .5

        # Step 3: 计算每个注意力头的平均距离
        # 使用张量乘法和求和操作计算注意力头的加权平均距离。
        # `attn_map` 表示每个注意力头在所有补丁对之间的权重分布。
        # `torch.einsum` 实现了加权求和操作，其中:
        # - `nm` 表示补丁之间的距离。
        # - `hnm` 表示注意力分布。
        # 结果为每个头的平均距离，形状为 [num_heads]。
        dist = torch.einsum('nm,hnm->h', (distances, attn_map))
        dist /= distances.size(0)

        # Step 4: 根据 `return_map` 参数决定返回内容
        # 如果 `return_map` 为 True，返回平均距离和完整的注意力分布图。
        # 否则，仅返回平均距离。
        if return_map:
            return dist, attn_map
        else:
            return dist

    def local_init(self, locality_strength=1.):
        """
            初始化局部位置编码，用于增强局部性。通过设置权重和偏置项，建立补丁间的相对位置信息。
            局部位置编码初始化后能够强化模型在空间上的局部感知能力。

            参数：
            - locality_strength (float): 控制局部增强的强度。数值越高，局部性效果越强。
        """
        # Step 1: 初始化 V 的权重矩阵为单位矩阵
        # `self.v` 是用于生成 Value 的线性层，将其权重初始化为单位矩阵，表示初始状态下每个输入通道与输出通道保持独立。
        # 这确保了初始状态下的特征保持不变。
        self.v.weight.data.copy_(torch.eye(self.dim))

        # Step 2: 设置局部距离的初始化参数
        # `locality_distance` 用于控制位置偏移的强度（默认为 1）。
        # 这个参数可以通过更复杂的公式（例如依赖 `locality_strength`）进一步调整。
        locality_distance = 1  # max(1, 1 / locality_strength ** .5)

        # Step 3: 计算注意力头的二维网格分布
        # 注意力头被视为二维网格上的节点，计算网格中心以建立局部位置偏移。
        kernel_size = int(self.num_heads ** .5)  # 假设注意力头可以在二维网格上排列。
        center = (kernel_size - 1) / 2 if kernel_size % 2 == 0 else kernel_size // 2  # 计算网格中心位置。

        # Step 4: 遍历网格中的每个位置，设置局部位置偏移
        for h1 in range(kernel_size):  # 遍历网格的第一维
            for h2 in range(kernel_size):  # 遍历网格的第二维
                position = h1 + kernel_size * h2  # 计算当前网格位置对应的头编号。

                # 设置相对位置编码的三个维度的初始权重：
                # - 第 0 维：水平方向的相对偏移，正负表示左右方向。
                self.pos_proj.weight.data[position, 2] = -1
                # - 第 1 维：垂直方向的相对偏移，正负表示上下方向。
                self.pos_proj.weight.data[position, 1] = 2 * (h1 - center) * locality_distance
                # - 第 2 维：与中心的距离平方，用于强调局部性。
                self.pos_proj.weight.data[position, 0] = 2 * (h2 - center) * locality_distance

        # Step 5: 调整权重值以结合局部增强强度
        # 通过乘以 `locality_strength`，控制位置编码对注意力的影响程度。
        self.pos_proj.weight.data *= locality_strength

    def get_rel_indices(self, num_patches):
        """
            计算补丁之间的相对位置索引（relative indices），用于描述补丁间的空间关系。
            这些相对位置索引为后续的局部性增强和相对位置编码提供了基础。

            参数：
            - num_patches (int): 图像被划分为的补丁数量，通常为 H × W（H 和 W 是图像补丁的宽高）。

            处理逻辑：
            - 每个补丁的位置索引用二维坐标表示。
            - 计算每两个补丁之间的水平偏移量 (x)、垂直偏移量 (y) 和距离平方 (x² + y²)。
            - 将这些相对位置信息存储在张量中。
        """
        # Step 1: 确定图像的补丁布局大小
        # 假设图像被均匀划分为 num_patches 个补丁，每个补丁为正方形。
        # 则 img_size 表示补丁网格的边长（假设为正方形）。
        img_size = int(num_patches ** .5)

        # 初始化相对位置索引张量，形状为 [1, num_patches, num_patches, 3]
        # - 第一维度为 batch size，占位用。
        # - 第二和第三维度表示补丁间的关系。
        # - 最后一维存储 3 个通道：水平偏移、垂直偏移、距离平方。
        rel_indices = torch.zeros(1, num_patches, num_patches, 3)

        # Step 2: 计算水平和垂直的相对偏移矩阵
        # `ind` 表示行列索引的相对偏移，例如：
        # - 对于 3×3 网格，ind = [[0, 1, 2], [1, 0, -1], [2, 1, 0]]。
        ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)

        # 计算每对补丁之间的水平和垂直偏移：
        # `indx` 表示水平偏移，重复网格大小的行。
        indx = ind.repeat(img_size, img_size)
        # `indy` 表示垂直偏移，重复网格大小的列。
        indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)

        # Step 3: 计算距离平方
        # 对每对补丁的水平和垂直偏移，计算距离平方。
        indd = indx ** 2 + indy ** 2

        # Step 4: 将计算结果存储到相对位置索引张量
        # 最后一维的 0 位置存储水平偏移。
        rel_indices[:, :, :, 2] = indd.unsqueeze(0)
        # 最后一维的 1 位置存储垂直偏移。
        rel_indices[:, :, :, 1] = indy.unsqueeze(0)
        # 最后一维的 2 位置存储距离平方。
        rel_indices[:, :, :, 0] = indx.unsqueeze(0)

        # Step 5: 将张量移动到与模型权重相同的设备
        # `self.qk.weight.device` 获取当前模型所在的计算设备（CPU 或 GPU）。
        device = self.qk.weight.device
        self.rel_indices = rel_indices.to(device)

# MHSA
class MHSA(nn.Module):
    """
        多头自注意力（Multi-Head Self-Attention, MHSA）模块：
        是 Transformer 中的核心组件，用于捕获序列（如图像补丁或词嵌入）之间的全局关系。
        支持多个注意力头，每个头专注于不同的特征子空间。

        structure:
        qkv(Linear) - Softmax - attn_drop(Dropout) - proj(Linear) - proj_drop(Dropout)
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        """
                初始化多头自注意力模块。

                参数：
                - dim (int): 输入特征的维度。
                - num_heads (int): 注意力头的数量，默认值为 8。
                - qkv_bias (bool): 是否为 Q、K、V 的线性层添加偏置项，默认值为 False。
                - qk_scale (float, 可选): 缩放因子。如果未指定，默认使用 `head_dim ** -0.5`。
                - attn_drop (float): 注意力权重的 Dropout 概率。
                - proj_drop (float): 输出特征的 Dropout 概率。
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads  # 每个注意力头的维度
        self.scale = qk_scale or head_dim ** -0.5  # Q 和 K 的缩放因子，防止点积结果过大。

        # 定义线性层，用于生成 Q、K、V。
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # Dropout 层，用于随机丢弃部分注意力权重，提升泛化能力。
        self.attn_drop = nn.Dropout(attn_drop)

        # 定义输出投影层，将多头注意力的结果映射回原特征维度。
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
                初始化模型权重。
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)  # 线性层权重初始化为截断正态分布
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)  # 偏置初始化为 0
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_attention_map(self, x, return_map=False):
        """
                计算注意力分布图和平均注意力距离。

                参数：
                - x (torch.Tensor): 输入张量，形状为 [batch_size, num_patches, dim]。
                - return_map (bool): 是否返回完整的注意力分布图，默认为 False。

                返回：
                - dist (torch.Tensor): 每个注意力头的平均距离，形状为 [num_heads]。
                - attn_map (torch.Tensor, 可选): 注意力分布图，形状为 [num_heads, num_patches, num_patches]。
        """
        B, N, C = x.shape  # B: 批量大小, N: 补丁数量, C: 特征维度

        # 通过线性层生成 Q、K、V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 计算注意力分布图
        attn_map = (q @ k.transpose(-2, -1)) * self.scale
        attn_map = attn_map.softmax(dim=-1).mean(0)  # 平均化 batch，得到 [num_heads, num_patches, num_patches]

        # 计算补丁之间的相对距离
        img_size = int(N ** .5)  # 假设补丁可以排列为正方形
        ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
        indx = ind.repeat(img_size, img_size)
        indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
        indd = indx ** 2 + indy ** 2
        distances = indd ** .5  # 补丁之间的实际距离
        distances = distances.to('cuda')

        # 计算注意力头的加权平均距离
        dist = torch.einsum('nm,hnm->h', (distances, attn_map))
        dist /= N

        if return_map:
            return dist, attn_map
        else:
            return dist

    def forward(self, x):
        """
                前向传播逻辑。

                参数：
                - x (torch.Tensor): 输入张量，形状为 [batch_size, num_patches, dim]。

                返回：
                - x (torch.Tensor): 输出张量，形状为 [batch_size, num_patches, dim]。
        """
        B, N, C = x.shape  # B: 批量大小, N: 补丁数量, C: 特征维度
        # Q、K、V 的形状分别为 [batch_size, num_heads, num_patches, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Step 2: 计算注意力权重
        # 点积注意力公式：softmax((Q @ K^T) / sqrt(d_k))
        attn = (q @ k.transpose(-2, -1)) * self.scale  # 形状为 [batch_size, num_heads, num_patches, num_patches]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)  # 应用 Dropout

        # Step 3: 加权 Value，计算输出
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # Step 4: 应用输出投影和 Dropout
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

# discriminator
class Discriminator(nn.Module):
    '''
    Discriminator block, used for patch-level adversarial training.

    structure:
        conv1(Conv2D) - act1(Relu) -
        conv2(Conv2D) - norm1(layerNorm) - act2(Relu) -
        conv3(Conv2D) - norm2(layerNorm) - act3(Relu) -
        conv4(Conv2D) - norm3(layerNorm) - act4(Relu) -
        conv5(Conv2D) - norm4(layerNorm) - act5(Relu) -
        flatten - output(Linear)
    '''
    def __init__(self, in_chans=3, num_classes=1, filter_size=4, num_filters=64, norm_layer=nn.LayerNorm, act_layer=nn.ReLU):
        '''
        初始化

        Args:
            :param in_chans (int): 输入通道数。默认为 3
            :param num_classes (int): 输出类别数。默认为 1
            :param filter_size (int): 卷积核大小。默认为 4
            :param num_filters (int): 卷积核数量。默认为 4
            :param norm_layer (nn.Module): 归一化层类。默认为 LayerNorm
            :param act_layer (nn.Module): 激活函数类。默认为 ReLU
        '''
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_chans, num_filters, kernel_size=filter_size, stride=2, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters * 2, kernel_size=filter_size, stride=2, padding=1)
        self.conv3 = nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=filter_size, stride=2, padding=1)
        self.conv4 = nn.Conv2d(num_filters * 4, num_filters * 8, kernel_size=filter_size, stride=2, padding=1)
        self.conv5 = nn.Conv2d(num_filters * 8, num_filters * 16, kernel_size=filter_size, stride=2, padding=1)

        # Normalization layers
        self.norm1 = norm_layer(num_filters)
        self.norm2 = norm_layer(num_filters*2)
        self.norm3 = norm_layer(num_filters*4)
        self.norm4 = norm_layer(num_filters*8)

        # Activation layers
        self.act1 = act_layer()
        self.act2 = act_layer()
        self.act3 = act_layer()
        self.act4 = act_layer()
        self.act5 = act_layer()

        # Output layer
        self.output = nn.Linear(num_filters*16, num_classes)

    def forward(self, x):
        '''
        forward

        Args:
            :param x: input tensor, shape [batch_size, in_chans, height, width]

        :return: output tensor, shape [batch_size, num_classes]
        '''

        x = self.conv1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.norm1(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.norm2(x)
        x = self.act3(x)

        x = self.conv4(x)
        x = self.norm3(x)
        x = self.act4(x)

        x = self.conv5(x)
        x = self.norm4(x)
        x = self.act5(x)

        x = x.flatten(1)
        x = self.output(x)

        return x