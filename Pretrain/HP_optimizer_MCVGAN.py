import numpy as np
import matplotlib.pyplot as plt
from Trainer_MCVGAN import Trainer_MCVGAN
from Model_MCVGAN import *
from datetime import datetime


class HP_optimizer_MCVGAN():
    '''MCVGAN 超参数优化器（基于遗传算法）'''
    def __init__(self, img_size=128, NP=60, select_ratio=0.8, L=18, G=20, Pc=0.8, Pm=0.05, train_mini_epochs=20):
        '''
        初始化超参数优化器
        :param NP: 种群数目
        :param select_ratio: 选择比例
        :param L: 染色体长度
        :param G: 进化代数
        :param Pc: 交叉概率
        :param Pm: 变异概率
        :param train_mini_epochs: 训练迭代次数

        染色体:
        0: lr = [1e-6, 1e-3)
        1: warmup_proportion = [1e-5, 1e-2)
        2: weight_decay = [1e-6, 1e-3)
        3: batch_size = (16, 32)
        4: embed_dim = (256, 512, 768, 1024)
        5: depth = [6, 48]
        6: num_heads = (8, 16, 32)
        7: mlp_ratio = (2, 4, 8)
        8: drop_rate = (0.1, 0.2, 0.3, 0.4, 0.5)
        9: attn_drop_rate = (0.1, 0.2, 0.3)
        10: drop_path_rate = (0.1, 0.2, 0.3)
        11: local_up_to_layer = (6, 8, 10, 12)
        12: locality_strength = (0.5, 1.0, 1.5)
        13: decoder_embed_dim = (256, 512, 768)
        14: decoder_depth = (4, 8, 12)
        15: decoder_num_heads = (4, 8, 16)
        16: filter_size = (3, 5, 7)
        17: num_filters = (32, 64, 128)
        '''
        self.img_size = img_size
        self.NP = NP
        self.select_ratio = select_ratio
        self.L = L
        self.G = G
        self.Pc = Pc
        self.Pm = Pm
        self.train_mini_epochs = train_mini_epochs

        # 初始化种群
        self.population = np.zeros((NP, L))
        for i in range(NP):
            self.population[i, 0] = np.random.uniform(1E-6, 1E-3)  # lr
            self.population[i, 1] = np.random.uniform(1E-5, 1E-2)  # warmup_proportion
            self.population[i, 2] = np.random.uniform(1E-6, 1E-3)    # weight_decay
            self.population[i, 3] = np.random.choice([16, 32], size=1)  # batch_size
            self.population[i, 4] = np.random.choice([256, 512, 768, 1024], size=1)  # embed_dim
            self.population[i, 5] = np.random.randint(6, 49)  # depth [6, 48]
            self.population[i, 6] = np.random.choice([8, 16, 32], size=1)  # num_heads
            self.population[i, 7] = np.random.choice([2, 4, 8], size=1)  # mlp_ratio
            self.population[i, 8] = np.random.choice([0.1, 0.2, 0.3, 0.4, 0.5], size=1)  # drop_rate
            self.population[i, 9] = np.random.choice([0.1, 0.2, 0.3], size=1)  # attn_drop_rate
            self.population[i, 10] = np.random.choice([0.1, 0.2, 0.3], size=1)  # drop_path_rate
            self.population[i, 11] = np.random.choice([6, 8, 10, 12], size=1)  # local_up_to_layer
            self.population[i, 12] = np.random.choice([0.5, 1.0, 1.5], size=1)  # locality_strength
            self.population[i, 13] = np.random.choice([256, 512, 768], size=1)  # decoder_embed_dim
            self.population[i, 14] = np.random.choice([4, 8, 12], size=1)  # decoder_depth
            self.population[i, 15] = np.random.choice([4, 8, 16], size=1)  # decoder_num_heads
            self.population[i, 16] = np.random.choice([3, 5], size=1)  # filter_size
            self.population[i, 17] = np.random.choice([32, 64, 128], size=1)  # num_filters

    def get_best_hyperparameters(self):
        '''
        获取最优超参数

        :return: 最优超参数组合
        '''
        # 日志记录
        with open("pretrain_log.txt", "a") as f:
            current_time = datetime.now()  # 获取当前日期和时间
            f.write(f"--------------------Start Hyperparameter optimize--------------------\n"
                    f"Start Time : {current_time}\n"
            )

        average_fitness_list = []  # 平均适应度列表
        best_fitness_list = []  # 最优适应度列表
        best_fitness = np.inf  # 最优适应度值
        x_best = None  # 最优个体
        count_gen = 0   # 记录优化代数

        # 进化迭代
        for gen in range(self.G):
            # 日志记录
            with open("pretrain_log.txt", "a") as f:
                current_time = datetime.now()
                f.write(f"\nGeneration: {gen + 1}\n"
                        f"Start Time : {current_time}\n"
                )

            # 计算适应度值
            fid_score = np.zeros((self.NP, 1))

            # deep learning
            for i in range(self.NP):
                lr = self.population[i, 0]
                warmup_proportion = self.population[i, 1]
                weight_decay = self.population[i, 2]
                batch_size = int(self.population[i, 3])
                embed_dim = int(self.population[i, 4])
                depth = int(self.population[i, 5])
                num_heads = int(self.population[i, 6])
                mlp_ratio = self.population[i, 7]
                drop_rate = self.population[i, 8]
                attn_drop_rate = self.population[i, 9]
                drop_path_rate = self.population[i, 10]
                local_up_to_layer = int(self.population[i, 11])
                locality_strength = self.population[i, 12]
                decoder_embed_dim = int(self.population[i, 13])
                decoder_depth = int(self.population[i, 14])
                decoder_num_heads = int(self.population[i, 15])
                filter_size = int(self.population[i, 16])
                num_filters = int(self.population[i, 17])

                # 清除显存缓存
                torch.cuda.empty_cache()
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')       # 使用 cuda
                generator = Masked_ConViT_GAN_Generator(
                    img_size=self.img_size,
                    embed_dim=embed_dim,
                    depth=depth,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=drop_path_rate,
                    local_up_to_layer=local_up_to_layer,
                    locality_strength=locality_strength,
                    decoder_embed_dim=decoder_embed_dim,
                    decoder_depth=decoder_depth,
                    decoder_num_heads=decoder_num_heads
                ).to(device)
                discriminator = Masked_ConViT_GAN_Discriminator(
                    img_size=self.img_size,
                    filter_size=filter_size,
                    num_filters=num_filters
                ).to(device)
                trainer = Trainer_MCVGAN(
                    generator=generator,
                    discriminator=discriminator,
                    lr=lr,
                    warmup_proportion=warmup_proportion,
                    weight_decay=weight_decay,
                    batch_size=batch_size,
                    img_size=self.img_size,
                    epochs=self.train_mini_epochs
                )

                fid = trainer.train_HP_optim(i)
                fid_score[i] = fid

            fitness = fid_score     # 以 FID score 作为适应度值

            # 记录平均适应度值
            average_fitness = np.mean(fitness)
            average_fitness_list.append(average_fitness)

            index = np.argmin(fitness)  # 最小值索引
            current_x_best = self.population[index].copy()  # 当代最优个体
            current_best_fitness = fitness[index].item()  # 当代最优适应度值

            # 记录当代最优适应度值
            best_fitness_list.append(current_best_fitness)
            # 记录日志：记录每一代平均适应度值、最优适应度值
            with open("pretrain_log.txt", "a") as f:
                f.write(f"\nAverage Fitness: {average_fitness:.4f}\n")
                f.write(f"Best Fitness: {current_best_fitness:.4f}\n")

            # 更新全局最优
            if current_best_fitness < best_fitness:
                x_best = current_x_best
                best_fitness = current_best_fitness

            # 归一化
            max_fitness = np.max(fitness)
            min_fitness = np.min(fitness)
            fitness_norm = (max_fitness - fitness) / (max_fitness - min_fitness)

            # 计算选择概率
            P = fitness_norm / np.sum(fitness_norm)
            P = P.flatten()  # 展平为一维

            # 选择（基于轮盘赌）
            selected_indices = np.random.choice(np.arange(self.NP), size=int(self.NP * self.select_ratio), replace=False, p=P)
            selected_population = self.population[selected_indices].copy()
            self.NP = selected_population.shape[0]  # 更新种群数目

            # 交叉
            for i in range(0, self.NP - 1, 2):
                if np.random.rand() < self.Pc:
                    # 随机选择交叉点
                    point = np.random.randint(1, self.L)
                    # 交叉
                    offspring1 = selected_population[i, point:].copy()
                    offspring2 = selected_population[i + 1, point:].copy()
                    selected_population[i, point:], selected_population[i + 1, point:] = offspring2, offspring1

            # 变异
            for i in range(self.NP):
                if np.random.rand() < self.Pm:
                    # 随机选择变异位
                    point = np.random.randint(0, self.L)
                    # 变异
                    if point == 0:  # lr 变异
                        selected_population[i, point] = np.random.uniform(1E-6, 1E-3)
                    elif point == 1:  # warmup_proportion 变异
                        selected_population[i, point] = np.random.uniform(1E-5, 1E-2)
                    elif point == 2:  # weight_decay 变异
                        selected_population[i, point] = np.random.uniform(1E-6, 1E-3)
                    elif point == 3:  # batch_size 变异
                        selected_population[i, point] = np.random.choice([16, 32], size=1)
                    elif point == 4:  # embed_dim 变异
                        selected_population[i, point] = np.random.choice([256, 512, 768, 1024], size=1)
                    elif point == 5:  # depth 变异
                        selected_population[i, point] = np.random.randint(6, 49)
                    elif point == 6:  # num_heads 变异
                        selected_population[i, point] = np.random.choice([8, 16, 32], size=1)
                    elif point == 7:  # mlp_ratio 变异
                        selected_population[i, point] = np.random.choice([2, 4, 8], size=1)
                    elif point == 8:  # drop_rate 变异
                        selected_population[i, point] = np.random.choice([0.1, 0.2, 0.3, 0.4, 0.5], size=1)
                    elif point == 9:  # attn_drop_rate 变异
                        selected_population[i, point] = np.random.choice([0.1, 0.2, 0.3], size=1)
                    elif point == 10:  # drop_path_rate 变异
                        selected_population[i, point] = np.random.choice([0.1, 0.2, 0.3], size=1)
                    elif point == 11:  # local_up_to_layer 变异
                        selected_population[i, point] = np.random.choice([6, 8, 10, 12], size=1)
                    elif point == 12:  # locality_strength 变异
                        selected_population[i, point] = np.random.choice([0.5, 1.0, 1.5], size=1)
                    elif point == 13:  # decoder_embed_dim 变异
                        selected_population[i, point] = np.random.choice([256, 512, 768], size=1)
                    elif point == 14:  # decoder_depth 变异
                        selected_population[i, point] = np.random.choice([4, 8, 12], size=1)
                    elif point == 15:  # decoder_num_heads 变异
                        selected_population[i, point] = np.random.choice([4, 8, 16], size=1)
                    elif point == 16:  # filter_size 变异
                        selected_population[i, point] = np.random.choice([3, 5, 7], size=1)
                    elif point == 17:  # num_filters 变异
                        selected_population[i, point] = np.random.choice([32, 64, 128], size=1)


            # 精英策略：将最优个体加入新种群
            reshaped_x_best = x_best.copy().reshape(1, self.L)
            new_population = np.append(selected_population, reshaped_x_best, axis=0)
            self.NP = new_population.shape[0]  # 更新种群数目

            # 更新种群
            self.population = new_population.copy()

            # 更新优化代数
            count_gen += 1

        # 输出结果

        # 将结果写入到文件中进行记录
        with open("pretrain_log.txt", "a") as f:
            f.write(f"\n最优适应度值: {best_fitness}\n"
                    "最优超参数:\n"
                    f"lr = {x_best[0]}\n"
                    f"warmup_proportion = {x_best[1]}\n"
                    f"weight_decay = {x_best[2]}\n"
                    f"batch_size = {x_best[3]}\n"
                    f"embed_dim = {x_best[4]}\n"
                    f"depth = {x_best[5]}\n"
                    f"num_heads = {x_best[6]}\n"
                    f"mlp_ratio = {x_best[7]}\n"
                    f"drop_rate = {x_best[8]}\n"
                    f"attn_drop_rate = {x_best[9]}\n"
                    f"drop_path_rate = {x_best[10]}\n"
                    f"local_up_to_layer = {x_best[11]}\n"
                    f"locality_strength = {x_best[12]}\n"
                    f"decoder_embed_dim = {x_best[13]}\n"
                    f"decoder_depth = {x_best[14]}\n"
                    f"decoder_num_heads = {x_best[15]}\n"
                    f"filter_size = {x_best[16]}\n"
                    f"num_filters = {x_best[17]}\n"
                    f"--------------------End--------------------\n"
            )

        print(f"最优适应度值: {best_fitness}\n"
              "最优超参数:\n"
              f"lr = {x_best[0]}\n"
              f"warmup_proportion = {x_best[1]}\n"
              f"weight_decay = {x_best[2]}\n"
              f"batch_size = {x_best[3]}\n"
              f"embed_dim = {x_best[4]}\n"
              f"depth = {x_best[5]}\n"
              f"num_heads = {x_best[6]}\n"
              f"mlp_ratio = {x_best[7]}\n"
              f"drop_rate = {x_best[8]}\n"
              f"attn_drop_rate = {x_best[9]}\n"
              f"drop_path_rate = {x_best[10]}\n"
              f"local_up_to_layer = {x_best[11]}\n"
              f"locality_strength = {x_best[12]}\n"
              f"decoder_embed_dim = {x_best[13]}\n"
              f"decoder_depth = {x_best[14]}\n"
              f"decoder_num_heads = {x_best[15]}\n"
              f"filter_size = {x_best[16]}\n"
              f"num_filters = {x_best[17]}\n"
        )

        # 绘制适应度曲线
        x = np.arange(start=1, stop=count_gen + 1)
        plt.plot(x, best_fitness_list, label='best', markevery=2)
        plt.plot(x, average_fitness_list, label='average', markevery=2)
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.ylim(0, 1.0)
        plt.legend(loc='lower right')
        plt.show()

        return x_best