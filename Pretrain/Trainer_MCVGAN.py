import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.image import FrechetInceptionDistance
from FaceMask_Dataset import FaceMask_Dataset_pretrain
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


class Trainer_MCVGAN():
    def __init__(self, generator, discriminator, lr, warmup_proportion, weight_decay, batch_size, img_size, epochs=15):
        '''
        初始化训练器

        Args:
            :param generator: 生成器
            :param discriminator: 判别器
            :param lr: 学习率
            :param warmup_proportion: 学习率预热比例
            :param weight_decay: 学习率衰减系数
            :param batch_size: 批次大小
            :param img_size: 图像大小
            :param train_mini_epochs: mini_train 迭代次数
            :param epochs: train 迭代次数
        '''
        self.generator = generator
        self.discriminator = discriminator
        self.lr = lr
        self.warmup_proportion = warmup_proportion
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.img_size = img_size
        self.epochs = epochs
        self.num_warmup_epochs = int(self.warmup_proportion * self.epochs)  # 预热轮数

        # 定义数据预处理
        transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),  # 调整图像大小
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # # 定义数据增强
        # transform_extend = transforms.Compose([
        #     transforms.Resize(img_size),  # 调整图像大小为 224 x 224
        #     transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0), ratio=(0.8, 1.25)),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])  # 图像标准化
        # ])

        # # 定义数据预处理(label)
        # transform_label = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Resize((224, 224)),  # 调整图像大小为 224 x 224
        # ])

        # 加载数据集
        train_dataset = FaceMask_Dataset_pretrain(img_dir="FaceMask/train", transform=transform)
        mini_train_dataset = FaceMask_Dataset_pretrain(img_dir='FaceMask/train_mini', transform=transform)
        # train_dataset_extend = FaceMask_Dataset(img_dir='FaceMask/train', transform=transform_extend)
        # train_dataset += train_dataset_extend
        validation_dataset = FaceMask_Dataset_pretrain(img_dir='FaceMask/validation', transform=transform)
        mini_validation_dataset = FaceMask_Dataset_pretrain(img_dir='FaceMask/validation_mini', transform=transform)
        test_dataset = FaceMask_Dataset_pretrain(img_dir='FaceMask/test', transform=transform)

        # 数据加载
        self.mini_train_loader = DataLoader(dataset=mini_train_dataset, batch_size=self.batch_size, shuffle=True)
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        self.validation_loader = DataLoader(dataset=validation_dataset, batch_size=self.batch_size, shuffle=False)
        self.mini_validation_loader = DataLoader(dataset=mini_validation_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False)

        # 使用 cuda
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 初始化模型
        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)

        # 定义损失函数
        self.loss_func_D = nn.BCEWithLogitsLoss()

        # 定义优化器
        self.optimizer_G = optim.AdamW(
            params=self.generator.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        self.optimizer_D = optim.AdamW(
            params=self.discriminator.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        # 定义 generator 学习率预热调度器
        self.lr_warmup_scheduler_G = optim.lr_scheduler.LambdaLR(
            self.optimizer_G,
            lr_lambda=self.lr_lambda
        )

        # 定义 generator 学习率衰减调度器
        self.lr_decay_scheduler_G = optim.lr_scheduler.StepLR(
            self.optimizer_G,
            step_size=30,
            gamma=self.weight_decay
        )

        # 定义 discriminator 学习率预热调度器
        self.lr_warmup_scheduler_D = optim.lr_scheduler.LambdaLR(
            self.optimizer_D,
            lr_lambda=self.lr_lambda
        )

        # 定义 discriminator 学习率衰减调度器
        self.lr_decay_scheduler_D = optim.lr_scheduler.StepLR(
            self.optimizer_D,
            step_size=30,
            gamma=self.weight_decay
        )

    # 学习率调整函数(预热)
    def lr_lambda(self, current_epoch):
        '''
        学习率调整函数(预热)
        :param current_epoch: 当前训练轮数
        :return:
        '''
        # 初期线性预热，从 0 开始到学习率
        if current_epoch < self.num_warmup_epochs:
            return float(current_epoch) / float(max(1, self.num_warmup_epochs))
        else:
            return 1        # 预热结束后，返回学习率不变

    # 训练(超参数优化)
    def train_HP_optim(self, index, k=10):
        '''
        训练
        :param index: 种群个体序号
        :param k: 判别器每 k step 训练一次
        :return: 最终 FID 分数
        '''
        # 记录日志
        with open("pretrain_log.txt", "a") as f:
            current_time = datetime.now()
            f.write(f"\nIndex: {index + 1}\n"
                    f"\n--------------------Start train-------------------\n"
                    f"Start time: {current_time}\n\n"
            )
        print(
            f"\nIndex: {index + 1}\n"
            f"img_size : {self.img_size}\n"
            f"lr : {self.lr}\n"
            f"weight_decay : {self.weight_decay}\n"
            f"warmup_proportion : {self.warmup_proportion}\n"
            f"batch_size : {self.batch_size}\n"
            f"embed_dim : {self.generator.embed_dim}\n"
            f"depth : {self.generator.depth}\n"
            f"num_heads : {self.generator.num_heads}\n"
            f"mlp_ratio : {self.generator.mlp_ratio}\n"
            f"drop_rate : {self.generator.drop_rate}\n"
            f"attn_drop_rate : {self.generator.attn_drop_rate}\n"
            f"drop_path_rate : {self.generator.drop_path_rate}\n"
            f"local_up_to_layer : {self.generator.local_up_to_layer}\n"
            f"locality_strength : {self.generator.locality_strength}\n"
            f"decoder_embed_dim : {self.generator.decoder_embed_dim}\n"
            f"decoder_depth : {self.generator.decoder_depth}\n"
            f"decoder_num_heads : {self.generator.decoder_num_heads}\n"
            f"filter_size : {self.discriminator.filter_size}\n"
            f"num_filters : {self.discriminator.num_filters}\n"
        )

        for epoch in range(1, self.epochs + 1):
            self.generator.train()
            self.discriminator.train()
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            epoch_mask_loss = 0.0
            epoch_extra_loss = 0.0
            count_train_d = 0

            for step, (X, y) in enumerate(self.mini_train_loader):
                X, y = X.to(self.device), y.to(self.device)
                batch_size = X.size(0)

                real_images = y
                real_labels = torch.ones(batch_size, 1).to(self.device)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)

                # 每 k 步训练一次判别器
                if step % k == 0:
                    # discriminator 对真实样本的 loss
                    outputs = self.discriminator(real_images)
                    d_loss_real = self.loss_func_D(outputs, real_labels)

                    # discriminator 对生成样本的 loss
                    g_loss, pred, _ = self.generator(X)
                    fake_images = self.generator.unpatchify(pred)
                    outputs = self.discriminator(fake_images.detach())
                    d_loss_fake = self.loss_func_D(outputs, fake_labels)

                    # discriminator 总 loss
                    d_loss = d_loss_real + d_loss_fake
                    epoch_d_loss += d_loss.item()
                    self.optimizer_D.zero_grad()
                    d_loss.backward()
                    self.optimizer_D.step()
                    self.lr_warmup_scheduler_D.step()
                    self.lr_decay_scheduler_D.step()

                    count_train_d += 1

                # 训练生成器
                mask_loss, pred, _ = self.generator(X)
                fake_images = self.generator.unpatchify(pred)
                outputs = self.discriminator(fake_images)
                extra_loss = self.loss_func_D(outputs, real_labels)
                g_loss = mask_loss + 0.5 * extra_loss
                epoch_g_loss += g_loss.item()
                epoch_mask_loss += mask_loss.item()
                epoch_extra_loss += extra_loss.item()
                self.optimizer_G.zero_grad()
                g_loss.backward()
                self.optimizer_G.step()
                self.lr_warmup_scheduler_G.step()
                self.lr_decay_scheduler_G.step()

                print(f"Epoch: {epoch}, Step: {step + 1}, G Loss: {g_loss.item():.4f}, D Loss: {d_loss.item():.4f}, mask loss: {mask_loss.item():.4f}, extra loss: {extra_loss.item():.4f}")

            average_G_loss = epoch_g_loss / len(self.mini_train_loader)
            average_D_loss = epoch_d_loss / count_train_d
            average_mask_loss = epoch_mask_loss / len(self.mini_train_loader)
            average_extra_loss = epoch_extra_loss / len(self.mini_train_loader)

            # 记录日志
            with open("pretrain_log.txt", "a") as f:
                f.write(f"Epoch: {epoch}, G Loss: {average_G_loss:.4f}(Mask Loss: {average_mask_loss:.4f}, Extra Loss: {average_extra_loss:.4f}), "
                  f"D Loss: {average_D_loss:.4f}\n"
                )

            print(f"Epoch: {epoch}, G Loss: {average_G_loss:.4f}(Mask Loss: {average_mask_loss:.4f}, Extra Loss: {average_extra_loss:.4f}), "
                  f"D Loss: {average_D_loss:.4f}")

        final_FID = self.get_fid_score(self.generator, self.mini_validation_loader, self.device)     # 最终 FID 分数

        print(f"Final FID: {final_FID:.4f}")

        # 记录日志
        with open("pretrain_log.txt", "a") as f:
            current_time = datetime.now()
            f.write(f"\nresult:\n"
                    f"Final FID: {final_FID:.4f}\n"
                    f"\nHyperparameters: \n"
                    f"img_size : {self.img_size}\n"
                    f"lr : {self.lr}\n"
                    f"weight_decay : {self.weight_decay}\n"
                    f"warmup_proportion : {self.warmup_proportion}\n"
                    f"batch_size : {self.batch_size}\n"
                    f"embed_dim : {self.generator.embed_dim}\n"
                    f"depth : {self.generator.depth}\n"
                    f"num_heads : {self.generator.num_heads}\n"
                    f"mlp_ratio : {self.generator.mlp_ratio}\n"
                    f"drop_rate : {self.generator.drop_rate}\n"
                    f"attn_drop_rate : {self.generator.attn_drop_rate}\n"
                    f"drop_path_rate : {self.generator.drop_path_rate}\n"
                    f"local_up_to_layer : {self.generator.local_up_to_layer}\n"
                    f"locality_strength : {self.generator.locality_strength}\n"
                    f"decoder_embed_dim : {self.generator.decoder_embed_dim}\n"
                    f"decoder_depth : {self.generator.decoder_depth}\n"
                    f"decoder_num_heads : {self.generator.decoder_num_heads}\n"
                    f"filter_size : {self.discriminator.filter_size}\n"
                    f"num_filters : {self.discriminator.num_filters}\n"
                    f"\nEnd time: {current_time}\n"
                    f"--------------------End--------------------\n"
            )

        return final_FID

    # 获取 FID 分数
    def get_fid_score(self, model, data_loader, device):
        '''
        获取 FID 分数

        Args:
            :param model: 模型
            :param data_loader: 数据加载器
            :param device: 设备
        :return: FID 分数
        '''
        fid_metric = FrechetInceptionDistance(feature=2048).to(device)
        model.eval()

        with torch.no_grad():
            for X, y in data_loader:
                X, y = X.to(device), y.to(device)
                _, pred, _ = model(X)
                fake_images = self.generator.unpatchify(pred)
                real_images = y
                fake_images = fake_images.to(torch.uint8)
                real_images = real_images.to(torch.uint8)
                fid_metric.update(fake_images, real=False)
                fid_metric.update(real_images, real=True)

        fid_score = fid_metric.compute().item()

        return fid_score

    # 训练
    def train(self, k=10):
        '''
        训练

        :param k: 判别器每 k step 训练一次
        '''
        # 记录日志
        with open("pretrain_log.txt", "a") as f:
            current_time = datetime.now()
            f.write(f"\n--------------------Start train-------------------\n"
                    f"Start time: {current_time}\n"
                    f"\nHyperparameters:\n"
                    f"img_size : {self.img_size}\n"
                    f"lr : {self.lr}\n"
                    f"weight_decay : {self.weight_decay}\n"
                    f"warmup_proportion : {self.warmup_proportion}\n"
                    f"batch_size : {self.batch_size}\n"
                    f"embed_dim : {self.generator.embed_dim}\n"
                    f"depth : {self.generator.depth}\n"
                    f"num_heads : {self.generator.num_heads}\n"
                    f"mlp_ratio : {self.generator.mlp_ratio}\n"
                    f"drop_rate : {self.generator.drop_rate}\n"
                    f"attn_drop_rate : {self.generator.attn_drop_rate}\n"
                    f"drop_path_rate : {self.generator.drop_path_rate}\n"
                    f"local_up_to_layer : {self.generator.local_up_to_layer}\n"
                    f"locality_strength : {self.generator.locality_strength}\n"
                    f"decoder_embed_dim : {self.generator.decoder_embed_dim}\n"
                    f"decoder_depth : {self.generator.decoder_depth}\n"
                    f"decoder_num_heads : {self.generator.decoder_num_heads}\n"
                    f"filter_size : {self.discriminator.filter_size}\n"
                    f"num_filters : {self.discriminator.num_filters}\n"
            )


        print(f"img_size : {self.img_size}\n"
              f"lr : {self.lr}\n"
              f"weight_decay : {self.weight_decay}\n"
              f"warmup_proportion : {self.warmup_proportion}\n"
              f"batch_size : {self.batch_size}\n"
              f"embed_dim : {self.generator.embed_dim}\n"
              f"depth : {self.generator.depth}\n"
              f"num_heads : {self.generator.num_heads}\n"
              f"mlp_ratio : {self.generator.mlp_ratio}\n"
              f"drop_rate : {self.generator.drop_rate}\n"
              f"attn_drop_rate : {self.generator.attn_drop_rate}\n"
              f"drop_path_rate : {self.generator.drop_path_rate}\n"
              f"local_up_to_layer : {self.generator.local_up_to_layer}\n"
              f"locality_strength : {self.generator.locality_strength}\n"
              f"decoder_embed_dim : {self.generator.decoder_embed_dim}\n"
              f"decoder_depth : {self.generator.decoder_depth}\n"
              f"decoder_num_heads : {self.generator.decoder_num_heads}\n"
              f"filter_size : {self.discriminator.filter_size}\n"
              f"num_filters : {self.discriminator.num_filters}\n"
        )

        train_fid_list = []
        val_fid_list = []

        for epoch in range(1, self.epochs + 1):
            self.generator.train()
            self.discriminator.train()
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            epoch_mask_loss = 0.0
            epoch_extra_loss = 0.0

            for step, (X, y) in enumerate(self.train_loader):
                X, y = X.to(self.device), y.to(self.device)
                batch_size = X.size(0)

                real_images = y
                real_labels = torch.ones(batch_size, 1).to(self.device)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)

                # 每 k 步训练一次判别器
                if step % k == 0:
                    # discriminator 对真实样本的 loss
                    outputs = self.discriminator(real_images)
                    d_loss_real = self.loss_func_D(outputs, real_labels)

                    # discriminator 对生成样本的 loss
                    g_loss, pred, _ = self.generator(X)
                    fake_images = self.generator.unpatchify(pred)
                    outputs = self.discriminator(fake_images.detach())
                    d_loss_fake = self.loss_func_D(outputs, fake_labels)

                    # discriminator 总 loss
                    d_loss = d_loss_real + d_loss_fake
                    epoch_d_loss += d_loss.item()
                    self.optimizer_D.zero_grad()
                    d_loss.backward()
                    self.optimizer_D.step()
                    self.lr_warmup_scheduler_D.step()
                    self.lr_decay_scheduler_D.step()

                # 训练生成器
                mask_loss, pred, _ = self.generator(X)
                fake_images = self.generator.unpatchify(pred)
                outputs = self.discriminator(fake_images)
                extra_loss = self.loss_func_D(outputs, real_labels)
                g_loss = mask_loss + 0.5 * extra_loss
                epoch_g_loss += g_loss.item()
                epoch_mask_loss += mask_loss.item()
                epoch_extra_loss += extra_loss.item()
                self.optimizer_G.zero_grad()
                g_loss.backward()
                self.optimizer_G.step()
                self.lr_warmup_scheduler_G.step()
                self.lr_decay_scheduler_G.step()

                # print(f"Epoch: {epoch}, Step: {step + 1}, G Loss: {g_loss.item():.4f}, D Loss: {d_loss.item():.4f}, mask loss: {mask_loss.item():.4f}, extra loss: {extra_loss.item():.4f}")

            average_G_loss = epoch_g_loss / len(self.train_loader)
            average_D_loss = epoch_d_loss / len(self.train_loader)
            average_mask_loss = epoch_mask_loss / len(self.train_loader)
            average_extra_loss = epoch_extra_loss / len(self.train_loader)

            train_FID = self.get_fid_score(self.generator, self.train_loader, self.device)
            train_fid_list.append(train_FID)
            val_FID = self.get_fid_score(self.generator, self.validation_loader, self.device)
            val_fid_list.append(val_FID)

            # 记录日志
            with open("pretrain_log.txt", "a") as f:
                f.write(f"Epoch: {epoch}, G Loss: {average_G_loss:.4f}(Mask Loss: {average_mask_loss:.4f}, Extra Loss: {average_extra_loss:.4f}), "
                        f"D Loss: {average_D_loss:.4f}, Train FID: {train_FID:.4f}, Val FID: {val_FID:.4f}\n"
                )

            print(f"Epoch: {epoch}, G Loss: {average_G_loss:.4f}(Mask Loss: {average_mask_loss:.4f}, Extra Loss: {average_extra_loss:.4f}), "
                  f"D Loss: {average_D_loss:.4f}, Train FID: {train_FID:.4f}, Val FID: {val_FID:.4f}")

        final_FID = self.get_fid_score(self.generator, self.validation_loader, self.device)     # 最终 FID 分数

        # 将结果写入到文件中进行记录
        with open("pretrain_log.txt", "a") as f:
            f.write(f"Final Val FID: {final_FID:.4f}\n")

        print(f"Final Val FID: {final_FID:.4f}")

        # 绘制 FID 分数曲线
        x = np.arange(self.epochs)
        plt.plot(x, train_fid_list, label='train', markevery=2)
        plt.plot(x, val_fid_list, label='val', markevery=2)
        plt.xlabel("epochs")
        plt.ylabel("FID")
        plt.legend(loc='lower right')
        plt.show()


    def save_generator(self):
        '''
        保存生成器模型参数
        '''
        torch.save(self.generator.state_dict(), "pretrain_models/best_pretrain_generator.pth")
        file_path = "pretrain_models/best_pretrain_generator.pth"
        print(f"Model parameters saved to {file_path}")



