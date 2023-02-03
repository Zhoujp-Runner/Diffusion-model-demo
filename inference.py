# _*_coding:utf-8_*_
# Author:Zhou JP
# DATE: 22:11 2023/2/3

import torch
from model import MLPDiffusion


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
print("all the shapes are same")

