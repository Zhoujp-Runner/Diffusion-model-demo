# _*_coding:utf-8_*_
# Author:Zhou JP
# DATE: 19:54 2023/2/3

import torch
import torch.nn as nn


class MLPDiffusion(nn.Module):

    def __init__(self, n_steps, num_unit=128):
        """
        初始化类
        :param num_unit: 隐藏层神经元的数量
        """
        super(MLPDiffusion, self).__init__()
        # 线性层组合
        self.linears = nn.ModuleList(
            [
                nn.Linear(2, num_unit),
                nn.ReLU(),
                nn.Linear(num_unit, num_unit),
                nn.ReLU(),
                nn.Linear(num_unit, num_unit),
                nn.ReLU(),
                nn.Linear(num_unit, 2)
            ]
        )
        # embedding层组合，不同深度上权重不一样，增加参数但不增加计算量
        self.step_embeddings = nn.ModuleList(
            [
                nn.Embedding(n_steps, num_unit),
                nn.Embedding(n_steps, num_unit),
                nn.Embedding(n_steps, num_unit)
            ]
        )

    def forward(self, x, t):
        """
        :param x: [batch_size, 2]
        :param t: [batch_size, 1]
        :return: [batch_size, 2]
        """
        for idx, embedding in enumerate(self.step_embeddings):
            t_embedding = embedding(t)  # [batch_size, 1, num_unit]
            t_embedding = t_embedding.squeeze(1)  # [batch_size, num_unit]
            x = self.linears[2 * idx](x)  # [batch_size, num_unit]
            x += t_embedding
            x = self.linears[2 * idx + 1](x)
        x = self.linears[-1](x)
        return x


if __name__ == '__main__':
    test = torch.IntTensor(128, 1)
    test = test.squeeze(-1)
    embedding = nn.Embedding(100, 128)
    print(embedding(test).shape)
