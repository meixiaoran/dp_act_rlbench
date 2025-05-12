import h5py

baddemo=[]
# with h5py.File("/home/rookie/桌面/diffusion_policy/data/dataset/close_laptop_lid/close_laptop_lid_merged_data.hdf5") as file:
#
#     # count total steps
#     demos = file['data']  # 打开 HDF5 文件并读取数据
#     episode_ends = list()
#     prev_end = 0
#     for i in range(179):
#         abort_variation = False
#         demo = demos[f'demo_{i}']
#         actions = demo['action']
#
#         if(demo['action'].shape[0] > 450):
#             baddemo.append(i)

print(baddemo)
print(len(baddemo))

import os

def clean_and_rename_files(folder_path, baddemo):
    """
    删除指定文件并重新排列剩余文件。
    
    Args:
        folder_path (str): 包含文件的目标文件夹路径。
        baddemo (list): 保存需要删除的文件编号的列表。
    """
    # 列出文件夹中所有的 .hdf5 文件
    files = [f for f in os.listdir(folder_path) if f.endswith('.hdf5')]
    
    # 按文件编号排序
    files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    # 遍历文件，删除 baddemo 中的文件
    for file in files:
        file_number = int(file.split('_')[1].split('.')[0])  # 提取文件编号
        if file_number in baddemo:
            os.remove(os.path.join(folder_path, file))  # 删除文件
            print(f"Deleted: {file}")

    # 列出剩余文件并重新排序
    remaining_files = [f for f in os.listdir(folder_path) if f.endswith('.hdf5')]
    remaining_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    # 按顺序重新命名文件
    for new_index, file in enumerate(remaining_files):
        new_name = f"episode_{new_index}.hdf5"
        old_path = os.path.join(folder_path, file)
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed: {file} -> {new_name}")

    print("Operation completed. Remaining files:")
    print([f for f in os.listdir(folder_path) if f.endswith('.hdf5')])


folder_path = "/home/rookie/桌面/diffusion_policy/RLBench_ACT/Datasets/wipe_desk/variation0"  # 替换为你的文件夹路径

# clean_and_rename_files(folder_path, baddemo)

    