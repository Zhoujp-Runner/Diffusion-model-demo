# _*_coding:utf-8_*_
# Author:Zhou JP
# DATE: 17:38 2023/2/3

# 数据集
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_s_curve
import torch


# 训练过程中用到的参数
num_steps = 100  # 扩散的步骤
betas = torch.linspace(-6, 6, num_steps)  # beta递增
betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5  # 约束beta的取值范围
alphas = 1 - betas  # 计算alpha(可能存在舍入误差)
alphas_prod = torch.cumprod(alphas, dim=0)  # 对alpha进行累乘，最后一个元素是所有alpha累乘的结果
alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]], 0)  # 让最后一项为t-1时刻的alpha_prod
alphas_bar_sqrt = torch.sqrt(alphas_prod)
one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)
assert alphas.shape == alphas_prod.shape == alphas_prod_p.shape == alphas_bar_sqrt.shape \
       == one_minus_alphas_bar_sqrt.shape == one_minus_alphas_bar_log.shape
print("all the shapes are same")


# 生成dataset
def get_dataset(num_point, viz=True):
    curve, _ = make_s_curve(num_point, noise=0.1)  # curve的类型是np.ndarry
    curve = curve[:, [0, 2]] / 10

    if viz:
        # 可视化数据集
        print("the shape of data: {}".format(np.shape(curve)))
        data = curve.T
        fig, ax = plt.subplots()
        ax.scatter(*data, color='red', edgecolor='white')
        ax.axis('off')
        plt.show()

    return torch.Tensor(curve).float()


# 扩散过程
def q_x(x_0, t):
    """
    基于重参数化技巧对扩散过程进行采样，即q(x_t|x_0)
    :param x_0: 原始数据
    :param t: 扩散的步骤
    :return: t时刻的采样值
    """
    noise = torch.randn_like(x_0)  # 返回一个大小和x_0一样的张量，里面元素是采样自标准正态分布
    alpha_t_bar_sqrt = alphas_bar_sqrt[t]
    one_minus_alpha_t_bar_sqrt = one_minus_alphas_bar_sqrt[t]
    return alpha_t_bar_sqrt * x_0 + one_minus_alpha_t_bar_sqrt * noise


# 可视化扩散过程
def viz_diffusion(x_0, num_show):
    """
    对扩散过程进行可视化
    :param x_0: 原始数据输入
    :param num_show: 需要展示的步骤数量，
    :return: None
    """
    fig, axs = plt.subplots(1, num_show, figsize=(28, 3))
    plt.rc('text', color='blue')
    for t in range(num_show):
        k = t % 10
        sample_t = q_x(x_0, torch.tensor([t * num_steps // num_show]))

        axs[k].scatter(sample_t[:, 0], sample_t[:, 1], color='red', edgecolor='white')
        axs[k].set_axis_off()
        axs[k].set_title('q(x_{})'.format(t * num_steps // num_show))
    plt.show()


if __name__ == '__main__':
    dataset = get_dataset(10**4, viz=False)
    viz_diffusion(dataset, 10)
