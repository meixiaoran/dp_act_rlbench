import torch
import random

def random_crop(inputs, crop_height, crop_width):
    batch_size, views, channels, height, width = inputs.shape
    
    # 随机选择裁剪区域的起始位置
    start_x = random.randint(0, height - crop_height)
    start_y = random.randint(0, width - crop_width)
    
    # 进行裁剪
    cropped_inputs = inputs[:, :, :, start_x:start_x+crop_height, start_y:start_y+crop_width]
    
    # 创建掩码：裁剪区域为 1，其他区域为 0
    mask = torch.zeros_like(inputs)
    mask[:, :, :, start_x:start_x+crop_height, start_y:start_y+crop_width] = 1

    # 创建一个新的张量，只有裁剪区域有值，其他区域为0
    expanded_inputs = torch.zeros_like(inputs)
    expanded_inputs[:, :, :, start_x:start_x+crop_height, start_y:start_y+crop_width] = cropped_inputs
    
    return cropped_inputs, expanded_inputs, mask

# 示例使用
inputs = torch.rand(64, 4, 3, 128, 128).cuda()  # 假设有一个 64 张图像的 batch
crop_height = 100
crop_width = 100

cropped_inputs, expanded_inputs, mask = random_crop(inputs, crop_height, crop_width)

# 输出结果的形状检查
print(cropped_inputs.shape)  # 应该是 (64, 4, 3, 100, 100)
print(expanded_inputs.shape)  # 应该是 (64, 4, 3, 128, 128)
print(mask.shape)            # 应该是 (64, 4, 3, 128, 128)

print(expanded_inputs[0][0][0][0])
