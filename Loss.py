# _*_coding:utf-8_*_
# Author:Zhou JP
# DATE: 20:39 2023/2/3

import torch
import torch.nn as nn


def diffusion_loss_fn(model, x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps):

    batch_size = x_0.shape[0]

    # 生成随机时刻t
    t = torch.randint(0, n_steps, size=(batch_size//2, ))  # 一半的t直接从标准正态分布中采样
    t = torch.cat([t, n_steps - 1 - t], dim=0)  # 另外一半的t用t_steps减去采样得到的一半，尽可能保证t遍布所有的时刻
    t = t.unsqueeze(-1)

    a = alphas_bar_sqrt[t]  # x_0的系数
    aml = one_minus_alphas_bar_sqrt[t]  # z（或者是eps）的系数
    e = torch.randn_like(x_0)  # 生成z或者是eps

    x = x_0 * a + aml * e  # 模型输入x_t

    out = model(x, t)  # 模型输出

    return (e - out).square().mean()


if __name__ == '__main__':
    # 生成随机时刻t
    t = torch.randint(0, 10, size=(10//2, ))  # 一半的t直接从标准正态分布中采样
    t = torch.cat([t, 10 - 1 - t], dim=0)  # 另外一半的t用t_steps减去采样得到的一半，尽可能保证t遍布所有的时刻
    print(t.shape)
    print(t)
    t = t.unsqueeze(-1)
