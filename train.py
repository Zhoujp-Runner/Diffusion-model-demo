# _*_coding:utf-8_*_
# Author:Zhou JP
# DATE: 21:01 2023/2/3

import torch
from sklearn.datasets import make_s_curve
import matplotlib.pyplot as plt

from model import MLPDiffusion
from Loss import diffusion_loss_fn


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


# 生成dataset
def get_dataset(num_point, viz=False):
    curve, _ = make_s_curve(num_point, noise=0.1)  # curve的类型是np.ndarry
    curve = curve[:, [0, 2]] / 10

    if viz:
        # 可视化数据集
        # print("the shape of data: {}".format(np.shape(curve)))
        data = curve.T
        fig, ax = plt.subplots()
        ax.scatter(*data, color='red', edgecolor='white')
        ax.axis('off')
        plt.show()

    return torch.Tensor(curve).float()


def train():

    dataset = get_dataset(10**4)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    fig, ax = plt.subplots()
    plt.rc('text', color='blue')

    model = MLPDiffusion(num_steps)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = 0

    for epoch in range(epochs):
        for idx, batch_x in enumerate(dataloader):
            loss = diffusion_loss_fn(model, batch_x, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, num_steps)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()

        if epoch % 200 == 0 or epoch == 3999:
            torch.save(model.state_dict(), 'D://model_save//diffusion_demo//diffusion_epoch_{}.pth'.format(epoch))
            print('Epoch {}: Loss is {}'.format(epoch, loss))
            print('Epoch {}: Model save done'.format(epoch))


if __name__ == '__main__':
    train()
