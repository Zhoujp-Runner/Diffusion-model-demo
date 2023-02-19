# _*_coding:utf-8_*_
# Author:Zhou JP
# DATE: 22:11 2023/2/3

import torch
from model import MLPDiffusion
import matplotlib.pyplot as plt


# 训练过程中用到的参数
batch_size = 128
epochs = 4000
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
print("all the shapes are same, shape is {}".format(alphas_prod.shape))

# 加载模型
epoch = 3999  # 需要加载第几个周期的模型
path = 'D://model_save//diffusion_demo//diffusion_epoch_{}.pth'.format(epoch)  # 模型加载的路径
model = MLPDiffusion(num_steps)  # 创建模型
model.load_state_dict(torch.load(path))  # 将模型参数加载进模型中


# 从原始序列中获取子序列
def get_s_steps(n_steps, stride):
    """

    :param n_steps: 原始序列的长度
    :param stride: 以stride为间隔，获得子序列
    :return: 子序列
    """
    return list(range(0, n_steps, stride))


def get_alpha_prod_st(s_steps, alpha_prod):
    """
    根据子序列，计算相应的alpha_prod
    :param s_steps: 子序列
    :param alpha_prod: 原始序列的alpha_prod
    :return: 子序列的alpha_prod_st
    """
    alpha_prod_st = dict()
    for i in s_steps:
        alpha_prod_st[i] = torch.tensor([alpha_prod[i]])
    return alpha_prod_st


def get_alpha_prod_p_st(s_steps, alpha_prod_st):
    alpha_prod_p_st = dict()
    st_1 = 0
    for i in s_steps:
        if i == 0:
            alpha_prod_p_st[i] = alpha_prod_st[0]
        else:
            alpha_prod_p_st[i] = alpha_prod_st[st_1]
            st_1 = i
    return alpha_prod_p_st


def get_beta_st(s_steps, alpha_prod_st):
    """

    :param s_steps: 子序列
    :param alpha_prod_st: 子序列对应的alpha_prod_st
    :return: beta_st
    """
    beta_st = dict()
    st_1 = 0
    for i in s_steps:
        if i == 0:
            beta_st[i] = torch.tensor([betas[0]])  # 先暂时让第0时刻的beta为0
        else:
            beta_st[i] = 1 - alpha_prod_st[i] / alpha_prod_st[st_1]
            st_1 = i
    return beta_st


# 正向推理
# 注意这里的seq中的数据含有grad，如果要对其进行计算，需要tensor.detach()
def diffusion_inference(diffusion_model, beta, alpha, alpha_prod, one_minus_alpha_bar_sqrt, alpha_prod_p, n_steps):
    # 生成T时刻的数据
    x_t = torch.randn(10**4, 2)
    seq = [x_t]
    # 进行正向推理，即由t推出t-1
    for t in reversed(range(n_steps)):
        x_t = sample(diffusion_model, x_t, t, beta, alpha, alpha_prod, one_minus_alpha_bar_sqrt, alpha_prod_p)
        seq.append(x_t)

    return seq


# 给定时刻t及其数据，采样t-1时刻的数据
def sample(diffusion_model, x_t, t, beta, alpha, alpha_prod, one_minus_alpha_bar_sqrt, alpha_prod_p):
    # 将整型t转化成张量
    t = torch.tensor([t])

    # 预测噪声
    z_sita = diffusion_model(x_t, t)

    # 计算t-1时刻分布的均值
    a = 1 / torch.sqrt(alpha[t])  # 总体的系数
    b = beta[t] / one_minus_alpha_bar_sqrt[t]  # 噪声的系数
    mean = a * (x_t - b * z_sita)

    # 计算t-1时刻的方差
    sigma = ((1 - alpha_prod_p[t]) / (1 - alpha_prod[t])) * beta[t]

    # 重参数化采样
    eps = torch.randn_like(x_t)  # 从标准正态分布中采样
    return mean + sigma * eps


# 加快采样速度
# 正向推理
# 注意这里的seq中的数据含有grad，如果要对其进行计算，需要tensor.detach()
def improve_diffusion_inference(diffusion_model, beta, alpha, alpha_prod, one_minus_alpha_bar_sqrt, alpha_prod_p, n_steps):
    # 生成T时刻的数据
    x_t = torch.randn(10**4, 2)
    seq = [x_t]

    s_steps = get_s_steps(n_steps, 2)  # 获得子序列st
    alpha_prod_st = get_alpha_prod_st(s_steps, alpha_prod)
    beta_st = get_beta_st(s_steps, alpha_prod_st)
    alpha_prod_p_st = get_alpha_prod_p_st(s_steps, alpha_prod_st)

    # 进行正向推理，即由st推出st-1
    for t in s_steps:
        x_t = improve_sample(diffusion_model, x_t, t, beta_st, alpha_prod_st, alpha_prod_p_st)
        seq.append(x_t)

    return seq


# 给定时刻st及其数据，采样st-1时刻的数据
def improve_sample(diffusion_model, x_t, t, beta, alpha_prod, alpha_prod_p):

    alpha = 1 - betas[t]
    a = 1 / torch.sqrt(alpha)  # 总体的系数
    one_minus_alpha_bar_sqrt = torch.sqrt(1 - alpha_prod[t])
    b = beta[t] / one_minus_alpha_bar_sqrt  # 噪声的系数

    # 计算t-1时刻的方差
    sigma = ((1 - alpha_prod_p[t]) / (1 - alpha_prod[t])) * beta[t]

    # 将整型t转化成张量
    t = torch.tensor([t])

    # 预测噪声
    z_sita = diffusion_model(x_t, t)
    mean = a * (x_t - b * z_sita)  # 计算t-1时刻分布的均值

    # 重参数化采样
    eps = torch.randn_like(x_t)  # 从标准正态分布中采样
    return mean + sigma * eps


# 可视化推理
def inference_viz(seq):
    fig, axs = plt.subplots(1, 11, figsize=(28, 3))
    plt.rc('text', color='blue')
    for idx, data in enumerate(seq):
        data = data.detach().numpy()  # data含有梯度，需要进行detach
        if idx % 10 == 0:
            col = idx // 10
            axs[col].scatter(data[:, 0], data[:, 1], color='red', edgecolor='white')
            axs[col].set_axis_off()
            axs[col].set_title('q(x_{})'.format(num_steps - idx))
    plt.show()


if __name__ == '__main__':
    seq = improve_diffusion_inference(model, betas, alphas, alphas_prod, one_minus_alphas_bar_sqrt, alphas_prod_p, num_steps)
    inference_viz(seq)
    # s_s = get_s_steps(num_steps, 2)
    # print(len(s_s))
    # alpha_st = get_alpha_prod_st(s_s, alphas_prod)
    # print(torch.tensor([alpha_st[0]]))
    # beta_st = get_beta_st(s_s, alpha_st)
    # print(beta_st)
    # alpha_prod_p_st = get_alpha_prod_p_st(s_s, alpha_st)
    # print(alpha_prod_p_st)
    # print(alpha_st)

