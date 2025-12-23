import torch
import torch.nn as nn
import torchvision
import numpy as np
from robomimic.models.base_nets import SpatialSoftmax

def get_resnet(name, input_shape, weights=None, **kwargs) -> nn.Module:
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", "r3m"
    """
    assert len(input_shape) == 3

    # 获取ResNet模型
    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)

    # 获取ResNet的卷积部分（去除最后的avgpool和fc层）
    resnet = nn.Sequential(*(list(resnet.children())[:-2]))

    net_list = [resnet]

    # 获取ResNet卷积层的输出维度
    x = torch.rand(1, input_shape[0], input_shape[1], input_shape[2])
    output_shape1 = resnet(x).shape
    output_shape1 = list(output_shape1)  # 转换为可修改的列表
    output_shape1.pop(0)  # 移除batch维度，变为(C, H, W)

    # 在ResNet的前面加上一个卷积层，将输入的通道数转换为3
    if input_shape[0] != 3:
        # 修改ResNet的第一层卷积，调整输入通道数
        # 原来的卷积层是：nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        # 改为适应64通道输入：nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3)
        resnet[0] = nn.Conv2d(input_shape[0], 64, kernel_size=7, stride=2, padding=3)
    
    # 加入SpatialSoftmax层
    pool = SpatialSoftmax(output_shape1, num_kp=32, learnable_temperature=False, temperature=1.0, noise_std=0.0)
    net_list.append(pool)

    # 获取SpatialSoftmax后的输出维度
    output_shape2 = pool(resnet(x)).shape

    # 加入Flatten层
    net_list.append(nn.Flatten(start_dim=1, end_dim=-1))

    # 加入一个线性层，输入维度是 output_shape2 所有元素的乘积
    linear = torch.nn.Linear(int(np.prod(output_shape2)), 1024)
    net_list.append(linear)

    # 将所有模块组合成一个Sequential模型
    net_list = nn.Sequential(*net_list)

    return net_list
