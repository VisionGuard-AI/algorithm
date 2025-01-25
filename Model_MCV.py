from Model_MCVAE import Masked_ConViT_Autoencoder
import torch.nn as nn
import torch

class Masked_ConViT(nn.Module):
    '''
    Masked Convolutional Vision Transformer for classification

    structure:

        Masked_ConViT_Autoencoder：
            patch_embed(PatchEmbed) - blocks(Block) - norm(LayerNorm)

        classifier(MLP)：
            Linear - ReLU - Linear
    '''
    def __init__(self, num_classes=8, hidden_dim=512, pretrained_MCVAE_path=None):
        '''
        初始化 Masked_ConViT

        Args：
            :param num_classes: 类别数
            :param hidden_dim: 分类器隐藏层维度
            :param pretrained_MCVAE_path: 预训练 Masked_ConViT_Autoencoder 模型路径
        '''
        super().__init__()

        self.num_classes = num_classes

        # 初始化 Masked_ConViT_Autoencoder
        self.encoder = Masked_ConViT_Autoencoder()
        # 加载预训练模型权重
        if pretrained_MCVAE_path:
            self.encoder.load_state_dict(torch.load(pretrained_MCVAE_path))

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )


    def forward(self, x):
        '''
        forward

        Args:
            :param x: 输入图像

        :return:
        '''
        # encoder
        latent = self.encoder(x)

        # classifier
        cls_output = latent[:, 0]       # 取出 class token
        logits = self.classifier(cls_output)

        return logits

