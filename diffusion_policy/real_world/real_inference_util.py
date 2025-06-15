from typing import Dict, Callable, Tuple
import numpy as np
from diffusion_policy.common.cv2_util import get_image_transform


def get_real_obs_dict(
        env_obs: Dict[str, np.ndarray],  # 环境中的观察数据，字典类型，键是观察数据的名称，值是对应的数据（如图像或低维数据）
        shape_meta: dict,  # 形状元数据，包含观察数据的形状和类型描述
) -> Dict[str, np.ndarray]:  # 返回一个字典，键是观察数据的名称，值是处理后的数据（图像或低维数据）

    obs_dict_np = dict()  # 初始化一个空字典，用于存储处理后的观察数据
    obs_shape_meta = shape_meta['obs']  # 获取观察数据的元数据，包括每个观察数据的形状和类型

    # 遍历每个观察数据的元数据
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')  # 获取数据的类型，默认值为 'low_dim'（低维数据）
        shape = attr.get('shape')  # 获取数据的形状信息

        # 处理类型为 'rgb' 的图像数据
        if type == 'rgb':
            this_imgs_in = env_obs[key]  # 获取对应的图像数据
            t, hi, wi, ci = this_imgs_in.shape  # 获取图像的维度信息：t=时间步，hi=高度，wi=宽度，ci=颜色通道数
            co, ho, wo = shape  # 获取目标图像的形状：co=目标颜色通道数，ho=目标高度，wo=目标宽度

            assert ci == co  # 确保输入图像和目标图像的颜色通道数相同

            out_imgs = this_imgs_in  # 初始的图像数据就是输入的图像数据

            # 如果目标图像的高度或宽度与输入图像不同，或者输入图像的类型是 uint8（可能需要归一化），
            # 则进行图像转换和尺寸调整
            if (ho != hi) or (wo != wi) or (this_imgs_in.dtype == np.uint8):
                # 获取一个图像变换函数，用于调整图像尺寸
                tf = get_image_transform(
                    input_res=(wi, hi),  # 输入图像的尺寸
                    output_res=(wo, ho),  # 输出图像的目标尺寸
                    bgr_to_rgb=False)  # 如果图像是 BGR 格式，需要转换为 RGB 格式

                # 对每一帧图像应用变换函数，调整图像尺寸
                out_imgs = np.stack([tf(x) for x in this_imgs_in])

                # 如果输入图像的像素值是 uint8（0-255），则归一化到 0-1 范围
                if this_imgs_in.dtype == np.uint8:
                    out_imgs = out_imgs.astype(np.float32) / 255

            # 将图像数据从 'THWC' 格式转换为 'TCHW' 格式
            # THWC 格式表示：时间步（T），高度（H），宽度（W），通道数（C）
            # TCHW 格式表示：时间步（T），通道数（C），高度（H），宽度（W）
            obs_dict_np[key] = np.moveaxis(out_imgs, -1, 1)

        # 处理类型为 'low_dim' 的低维数据
        elif type == 'low_dim':
            this_data_in = env_obs[key]  # 获取对应的低维数据

            # 如果数据包含 "pose" 且形状为 (2,)，说明是坐标数据，只取前两个维度（X, Y）
            if 'pose' in key and shape == (2,):
                this_data_in = this_data_in[..., [0, 1]]  # 只取 X 和 Y 坐标

            # 将低维数据直接加入到返回字典中
            obs_dict_np[key] = this_data_in

    # 返回处理后的观察数据字典
    return obs_dict_np


def get_real_obs_resolution(
        shape_meta: dict
        ) -> Tuple[int, int]:
    out_res = None
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        if type == 'rgb':
            co,ho,wo = shape
            if out_res is None:
                out_res = (wo, ho)
            assert out_res == (wo, ho)
    return out_res
