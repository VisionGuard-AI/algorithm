from Pretrain.Blocks import *
from utils.pos_embed import *

class Masked_ConViT_Autoencoder(nn.Module):
    '''
    Masked Convolutional Vision Transformer Autoencoder

    structure:
        patch_embed(PatchEmbed) - blocks(Block) - norm(LayerNorm)
    '''
    def __init__(self, img_size=128, patch_size=16, in_chans=3, num_classes=1, embed_dim=1024, depth=24,
                 num_heads=16, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, local_up_to_layer=10, locality_strength=1.,
                 use_pos_embed=True, norm_pix_loss=False):
        '''
        初始化 Masked_ConViT_Autoencoder

        Args：
            :param img_size: 图像大小
            :param patch_size: 每个 patch 的大小
            :param in_chans: 输入通道数
            :param num_classes: 类别数
            :param embed_dim: 每个 patch 的嵌入维度
            :param depth: Transformer 块的层数
            :param num_heads: 注意力头的数量(应为 2 的幂次)
            :param mlp_ratio: MLP 中隐藏层维度与嵌入维度的比例
            :param qkv_bias: 是否为 Q、K、V 的线性层添加偏置项
            :param qk_scale: Q 和 K 的缩放因子
            :param drop_rate: Dropout 概率
            :param attn_drop_rate: 注意力权重的 Dropout 概率
            :param drop_path_rate: 随机深度的丢弃概率
            :param norm_layer: 归一化层类，默认使用 LayerNorm
            :param local_up_to_layer: 在指定层之前使用局部增强注意力（GPSA），之后使用标准 MHSA
            :param locality_strength: GPSA 的局部增强强度
            :param use_pos_embed: 是否使用位置嵌入
            :param norm_pix_loss: 是否对像素损失进行归一化
        '''
        super().__init__()

        self.num_classes = num_classes
        self.local_up_to_layer = local_up_to_layer
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.locality_strength = locality_strength
        self.use_pos_embed = use_pos_embed
        self.patch_size = patch_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.local_up_to_layer = local_up_to_layer
        self.locality_strength = locality_strength

        # encoder

        # 初始化 Patch Embed
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        # 计算 patches 总数
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches

        # 初始化 class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 初始化 position dropout
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 初始化 position embed
        if self.use_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        # 初始化 Transformer blocks

        # 定义随机深度的丢弃规则
        drop_rule = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        # transformer block list
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_rule[i], norm_layer=norm_layer,
                use_gpsa=True, locality_strength=locality_strength
            )
            if i < local_up_to_layer else
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_rule[i], norm_layer=norm_layer,
                use_gpsa=False
            )
            for i in range(depth)
        ])

        # 初始化 norm
        self.norm = norm_layer(embed_dim)

        # 是否归一化 pixel loss
        self.norm_pix_loss = norm_pix_loss

        # 初始化权重
        self.initialize_weights()

    def initialize_weights(self):
        '''
        初始化权重
        '''
        # 初始化 class_token 权重
        trunc_normal_(self.class_token, std=.02)

        # 初始化 pos_embed 权重
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # 初始化 patch_embed 权重
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))  # 使用 Xavier 均匀初始化方法

        self.apply(self._init_weights)

    def _init_weights(self, m):
        '''
        初始化权重

        Args：
            :param m: 模块

        '''
        if isinstance(m, nn.Linear):        # 判断当前模块是否是线性层 nn.Linear
            torch.nn.init.xavier_uniform_(m.weight)
            # 如果线性层有偏置项，则将偏置项初始化为零
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        # 判断当前模块是否是归一化层 nn.LayerNorm
        elif isinstance(m, nn.LayerNorm):
            # 将 LayerNorm 层的偏置项初始化为零
            nn.init.constant_(m.bias, 0)
            # 将 LayerNorm 层的缩放因子（权重）初始化为 1.0
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        '''
        将输入图像转换为 patches

        Args:
            :param imgs: 输入图像

        :return: patches
        '''
        # 获取 patch size
        p = self.patch_embed.patch_size

        # 校验图像的宽高是否能整除 patch size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        # 计算每个图像的高和宽将被切割成多少个 patch
        h = w = imgs.shape[2] // p

        # 将输入的图像按 Patch Size 切割成多个块(每个图像按通道数、patch 高度和宽度分割成块)
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))

        # 维度重排，图像的形状变为 [N, h, w, p, p, 3]，每个 patch 形成一个 (p, p, 3) 的块
        x = torch.einsum('nchpwq->nhwpqc', x)

        # 将形状变为[ N, (h * w), (p^2 * 3) ], 表示每张图像有 h * w 个 patch，每个 patch 是一个展平的向量。
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))

        return x

    def unpatchify(self, x):
        '''
        将 patches 转换为图像
        Args:
            :param x: patches
        :return: 图像
        '''
        # patch size
        p = self.patch_embed.patch_size

        # 计算恢复后的图像高和宽
        h = w = int(x.shape[1] ** .5)

        # 校验恢复后的 patch 数量是否正确
        assert h * w == x.shape[1]

        # 将输入数据恢复成一个形状为 (N, h, w, p, p, 3) 的张量
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))

        # 维度重排，转换回 (N, 3, h, p, w, p) 的形状
        x = torch.einsum('nhwpqc->nchpwq', x)

        # 将恢复后的图像展平成原始的图像形状 (N, 3, h * p, h * p)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))

        return imgs

    def random_masking(self, x, mask_ratio):
        '''
        随机掩码

        Args:
            :param x: patches [N, L, D]
            :param mask_ratio: 掩码比例

        :return:
            x_masked: 掩码后的 patches
            mask: 掩码
            ids_restore: 掩码还原索引
        '''
        # 获取 x 的形状
        N, L, D = x.shape  # batch, length, dim

        # 计算保留的部分的长度
        len_keep = int(L * (1 - mask_ratio))

        # 生成 [0, 1] 区间的随机噪声，形状为 (N, L)，每个样本和位置对应一个噪声值
        noise = torch.rand(N, L, device=x.device)

        # 对每个样本的噪声进行排序，返回排序后的索引
        # 按照噪声从小到大排序（较小的噪声对应的 patch 会被保留，较大的噪声对应的 patch 会被移除）
        ids_shuffle = torch.argsort(noise, dim=1)

        # 根据排序后的索引恢复原始顺序，用于后续还原掩蔽的位置
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # 选择排序后前 len_keep 个索引（表示需要保留的 patch）
        ids_keep = ids_shuffle[:, :len_keep]

        # 获取被保留的部分数据
        # 将原始序列 `x` 根据 `ids_keep` 索引提取出前 len_keep 个元素（被保留部分）
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # 生成 mask，0 表示保留，1 表示移除
        mask = torch.ones([N, L], device=x.device)

        mask[:, :len_keep] = 0      # 将前 len_keep 个位置设为 0，表示这些位置是保留的

        # 根据恢复的顺序（`ids_restore`）将掩蔽还原到原始顺序
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_feature(self, imgs, mask_ratio=0.75):
        '''
        encoder forward

        Args:
            :param imgs: 输入图像
            :param mask_ratio: 掩码比例
        :return:
            x: image embedding(latent)
            mask: 掩码
            ids_restore: 掩码还原索引
        '''
        # 获取批量大小
        B = imgs.shape[0]

        # embed patches
        x = self.patch_embed(imgs)

        # add position embedding
        if self.use_pos_embed:
            x = x + self.pos_embed
        x = self.pos_drop(x)        # 应用 Dropout

        # 随机掩码
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        class_tokens = self.class_token.expand(B, -1, -1)       # [batch_size, 1, embed_dim]

        # blocks forward
        for u, block in enumerate(self.blocks):
            if u == self.local_up_to_layer:
                # add class token，形状变为 [batch_size, num_patches + 1, embed_dim]
                x = torch.cat((class_tokens, x), dim=1)
            x = block(x)  # 通过当前 Transformer 块处理

        # normalize
        x = self.norm(x)

        return x, mask, ids_restore

    def forward(self, imgs, mask_ratio=0.75):
        '''
        forward

        Args:
            :param imgs: 输入图像
            :param mask_ratio: 掩码比例
        :return:
            latent: image embedding
            mask: 掩码
        '''
        # encoder
        latent, _, _ = self.forward_feature(imgs, mask_ratio=mask_ratio)

        return latent
