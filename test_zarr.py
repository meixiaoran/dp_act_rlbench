import zarr
from zarr.hierarchy import Group

def inspect_zarr_file(file_path):
    # 打开Zarr文件
    store = zarr.open(file_path, mode='r')  # 以只读模式打开Zarr文件

    # 打印Zarr文件的根目录内容（文件结构）
    print("Zarr file structure:")
    print(store)

    # 列出所有数组和组
    print("\nArrays and datasets in the Zarr file:")
    for name in store:
        print(name)

    # 查看每个数组或组的基本信息
    print("\nDetails of arrays or groups:")
    for name in store:
        item = store[name]

        print(f"Group: {name}")
        print(f"  Group contains: {list(item)}")

        # 如果是 'data' 组，进一步检查里面的 'action', 'head', 'qpos', 'wrist' 数据集
        if name == 'data':
            if 'action' in item:
                action_array = item['action']
                print(f"Action Array Shape: {action_array.shape}")
                # print(action_array[:10])
            if 'qpos' in item:
                qpos_array = item['qpos']
                print(f"Qpos Array Shape: {qpos_array.shape}")
                # print(qpos_array[:10])



        # 如果是 'meta' 组，进一步检查里面的 'episode_ends' 数据集
        elif name == 'meta' and 'episode_ends' in item:
            episode_ends_array = item['episode_ends']
            print(f"Episode Ends Array Shape: {episode_ends_array.shape}")

if __name__ == "__main__":
    # 替换成你自己的Zarr文件路径
    file_path = "/home/mar/下载/slide_block_to_target_merged_data.hdf5.zarr.zip"
    inspect_zarr_file(file_path)
